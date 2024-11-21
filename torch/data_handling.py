# Imports
import chess
import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
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
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

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


def parse_pgn_to_tensors_parallel_with_limited_resources(pgn_files, output_file, batch_size=1000, num_workers=None):
    """
    Processes PGN files in parallel with controlled resource usage, writes tensors and moves to an HDF5 file incrementally.
    """
    # Limit the number of worker processes
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)  # Use half of available CPUs

    with h5py.File(output_file, "w") as h5file:
        tensor_group = h5file.create_group("tensors")
        moves_group = h5file.create_group("moves")

        batch_count = 0  # Track the number of batches written

        # Parallel processing with a progress bar for files
        with tqdm(total=len(pgn_files), desc="Processing PGN Files") as file_bar:
            with Pool(processes=num_workers) as pool:
                for board_tensors, next_moves in pool.imap(process_pgn_file_with_progress, pgn_files):
                    # Process data in smaller chunks
                    for i in range(0, len(board_tensors), batch_size):
                        tensor_batch = batch_to_numpy(board_tensors[i:i + batch_size])
                        move_batch = np.array(next_moves[i:i + batch_size], dtype=np.int32)

                        # Write the batch to HDF5
                        tensor_group.create_dataset(
                            f"batch_{batch_count}",
                            data=tensor_batch,
                            compression="gzip",
                            chunks=True,
                        )
                        moves_group.create_dataset(
                            f"batch_{batch_count}",
                            data=move_batch,
                            compression="gzip",
                            chunks=True,
                        )

                        batch_count += 1

                        # Flush data to disk after each batch
                        h5file.flush()

                    file_bar.update(1)  # Update the file progress bar

    print(f"Processed {batch_count} batches into {output_file}")



def batch_to_numpy(tensor_list):
    """Convert a list of PyTorch tensors to a single numpy array."""
    return np.stack([tensor.numpy() for tensor in tensor_list])


class ChessDataset(Dataset):
    def __init__(self, hdf5_file):
        """Initialize the dataset and index all board states."""
        self.hdf5_file = hdf5_file
        self.sample_index = []  # Map global sample index to (key, index in batch)
        
        with h5py.File(hdf5_file, "r") as h5file:
            for key in sorted(h5file["tensors"].keys()):
                num_samples = h5file["tensors"][key].shape[0]  # Number of samples in this batch
                self.sample_index.extend([(key, i) for i in range(num_samples)])  # Add (key, index) pairs
                
        self.length = len(self.sample_index)  # Total number of samples

    def __len__(self):
        """Return the total number of board states."""
        return self.length

    def __getitem__(self, idx):
        """Load a specific board state on demand."""
        key, local_idx = self.sample_index[idx]  # Map global index to (key, index)
        
        with h5py.File(self.hdf5_file, "r") as h5file:
            tensor = h5file["tensors"][key][local_idx]  # Access specific board state
            move = h5file["moves"][key][local_idx]      # Access corresponding move

        # Convert to PyTorch tensors
        tensor = torch.from_numpy(tensor).float()
        move = torch.from_numpy(move).long()

        return tensor, move


if __name__ == "__main__":
    # Step 1: Import PGN file paths
    pgn_files = import_data(25, 30)

    # Step 2: Convert PGN files to tensors and store them in HDF5
    output_file = "chess_data.hdf5"
    parse_pgn_to_tensors_parallel_with_limited_resources(pgn_files, output_file)

    # Step 3: Load dataset and verify
    dataset = ChessDataset(output_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    with h5py.File("chess_data.hdf5", "r") as h5file:
        print("Tensors:", list(h5file["tensors"].keys()))
        print("Moves:", list(h5file["moves"].keys()))

    for i, (tensor, move) in enumerate(dataloader):
        print(f"Batch {i}: Tensor shape = {tensor.shape}, Move shape = {move.shape}")
        if i == 2:  # Only process a few batches for verification
            break
