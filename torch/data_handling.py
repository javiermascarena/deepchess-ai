# Imports
import chess
import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import h5py  # For efficient data storage
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping of piece types to tensor indices
PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def import_data(start_idx=0, end_idx=79) -> list:
    """
    Returns a list of PGN file paths within the specified range.
    
    Parameters:
        start_idx (int): The index of the first file to include.
        end_idx (int): The index of the last file (exclusive) to include.
    
    Returns:
        list: A list of PGN file paths.
    """
    # Getting the absolute path
    data_relative_path = os.path.join(".", "chess-data", "pgn")
    data_absolute_path = os.path.abspath(data_relative_path)
    data = []

    # Enumerate files, only keep those within the range
    for i, file_name in enumerate(os.listdir(data_absolute_path)):
        if i < start_idx:
            continue
        if i >= end_idx:
            break
        file_path = os.path.join(data_absolute_path, file_name)
        if os.path.isfile(file_path):
            data.append(file_path)

    return data


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Returns a 12x8x8 tensor for the board, with one layer per piece type."""
    tensor = torch.zeros((12, 8, 8), dtype=torch.float16)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_idx = PIECE_MAP[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            tensor[piece_idx + color_offset, row, col] = 1

    return tensor


def process_pgn_file_with_progress(pgn_file_path):
    """Processes a single PGN file with progress tracking."""
    board_tensors = []
    next_moves = []

    with open(pgn_file_path) as pgn_file:
        total_games = sum(1 for _ in chess.pgn.read_headers(pgn_file))  # Count total games
        pgn_file.seek(0)  # Reset file pointer to start
        progress_bar = tqdm(total=total_games, desc=f"Processing {os.path.basename(pgn_file_path)}")

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            tensor = board_to_tensor(board)  # Create initial tensor

            for move in game.mainline_moves():
                board_tensors.append(tensor.clone())
                next_moves.append((move.from_square, move.to_square))
                board.push(move)  # Apply move to the board

            progress_bar.update(1)  # Update progress for each game

        progress_bar.close()

    return board_tensors, next_moves


def parse_pgn_to_tensors_with_optimized_hdf5(pgn_files, output_file, batch_size=1000):
    """
    Optimized PGN-to-tensors conversion with memory-efficient HDF5 writing.
    """
    with h5py.File(output_file, "w") as h5file:
        # Create extendable datasets
        tensor_dataset = h5file.create_dataset(
            "tensors",
            shape=(0, 12, 8, 8),
            maxshape=(None, 12, 8, 8),
            dtype='float16',
            compression="gzip",
            chunks=True,
        )
        moves_dataset = h5file.create_dataset(
            "moves",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype='int16',
            compression="gzip",
            chunks=True,
        )

        tensor_offset = 0  # Tracks the current write position
        move_offset = 0

        # Process each file
        for pgn_file in tqdm(pgn_files, desc="Processing PGN Files"):
            board_tensors, next_moves = process_pgn_file_with_progress(pgn_file)
            
            # Convert to numpy arrays for efficient appending
            tensor_batch = batch_to_numpy(board_tensors)
            move_batch = np.array(next_moves, dtype=np.int16)

            # Resize datasets to accommodate the new batch
            new_tensor_size = tensor_offset + tensor_batch.shape[0]
            new_move_size = move_offset + move_batch.shape[0]

            tensor_dataset.resize(new_tensor_size, axis=0)
            moves_dataset.resize(new_move_size, axis=0)

            # Append the new data
            tensor_dataset[tensor_offset:new_tensor_size] = tensor_batch
            moves_dataset[move_offset:new_move_size] = move_batch

            tensor_offset = new_tensor_size
            move_offset = new_move_size

            # Flush changes to disk
            h5file.flush()

    print(f"Processed tensors and moves saved to {output_file}")



def batch_to_numpy(tensor_list):
    """Convert a list of PyTorch tensors to a single numpy array."""
    return np.stack([tensor.numpy() for tensor in tensor_list])


def inspect_hdf5_file(hdf5_file, sample_count=5):
    """
    Inspects the contents of an HDF5 file and prints samples for verification.
    """
    with h5py.File(hdf5_file, "r") as h5file:
        # Print dataset shapes
        print("Tensors dataset shape:", h5file["tensors"].shape)
        print("Moves dataset shape:", h5file["moves"].shape)

        # Sample data from the start
        print("\nSample tensors:")
        for i in range(sample_count):
            print(h5file["tensors"][i])
        
        print("\nSample moves:")
        for i in range(sample_count):
            print(h5file["moves"][i])


def clear_hdf5_file(hdf5_file):
    """
    Clears all datasets in the HDF5 file by deleting and recreating it.
    """
    with h5py.File(hdf5_file, "w") as h5file:
        # Recreate empty datasets
        h5file.create_dataset(
            "tensors",
            shape=(0, 12, 8, 8),
            maxshape=(None, 12, 8, 8),
            dtype='float32',
            compression="gzip",
            chunks=True,
        )
        h5file.create_dataset(
            "moves",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype='int32',
            compression="gzip",
            chunks=True,
        )
    print(f"{hdf5_file} has been cleared.")


class ChessDataset(Dataset):
    def __init__(self, hdf5_file):
        """Initialize the dataset and index all board states."""
        self.hdf5_file = hdf5_file
        
        with h5py.File(hdf5_file, "r") as h5file:
            # Use dataset lengths directly
            self.num_samples = h5file["tensors"].shape[0]

    def __len__(self):
        """Return the total number of board states."""
        return self.num_samples

    def __getitem__(self, idx):
        """Load a specific board state on demand."""
        with h5py.File(self.hdf5_file, "r") as h5file:
            # Access data by index
            tensor = h5file["tensors"][idx]
            move = h5file["moves"][idx]

        # Convert to PyTorch tensors
        tensor = torch.from_numpy(tensor).float()
        move = torch.from_numpy(move).long()

        return tensor, move



if __name__ == "__main__":
     # Step 1: Clear the file before processing
    output_file = "chess_data.hdf5"
    clear_hdf5_file(output_file)
    
    # Step 2: Import all PGN files
    pgn_files = import_data(0, 45)

    # Step 3: Convert all PGN files to tensors and store them in HDF5
    parse_pgn_to_tensors_with_optimized_hdf5(pgn_files, output_file)

    # Step 4: Verify the contents of the file
    inspect_hdf5_file(output_file, sample_count=5)