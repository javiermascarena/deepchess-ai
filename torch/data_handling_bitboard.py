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


def generate_all_possible_moves():
    """Generate a global list of all valid chess moves."""
    all_moves = []
    board = chess.Board()

    # Iterate over all possible squares for 'from' and 'to'
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Add standard moves
            move = chess.Move(from_square, to_square)
            if board.is_legal(move):
                all_moves.append(move)

            # Add promotion moves (only valid for pawns moving to the last rank)
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion)
                if board.is_legal(move):
                    all_moves.append(move)

    return all_moves

# Step 1: Create global move-to-index mapping
ALL_MOVES = generate_all_possible_moves()
MOVE_TO_INDEX = {move: idx for idx, move in enumerate(ALL_MOVES)}

def extract_bitboards(data: list):
    bitboards = []
    moves = []

    for pgn_file_path in data: 

        with open(pgn_file_path) as pgn_file: 
            while True: 
                game = chess.pgn.read_game(pgn_file)

                if game is None: 
                    break

                board = game.board()
                game_bitboards = []
                game_moves = []

                for move in game.mainline_moves():
                    bitboard = board.occupied
                    game_bitboards.append(bitboard)

                    move_idx = MOVE_TO_INDEX.get(move)
                    game_moves.append(move_idx)

                    board.push(move)

                bitboards.append(game_bitboards)
                moves.append(game_moves)

    return bitboards, moves


"""class ChessDataset(Dataset):
    def __init__(self, hdf5_file):
        Initialize the dataset and index all board states.
        self.hdf5_file = hdf5_file
        
        with h5py.File(hdf5_file, "r") as h5file:
            # Use dataset lengths directly
            self.num_samples = h5file["tensors"].shape[0]

    def __len__(self):
        Return the total number of board states.
        return self.num_samples

    def __getitem__(self, idx):
        Load a specific board state on demand.
        with h5py.File(self.hdf5_file, "r") as h5file:
            # Access data by index
            tensor = h5file["tensors"][idx]
            move = h5file["moves"][idx]

        # Convert to PyTorch tensors
        tensor = torch.from_numpy(tensor).float()
        move = torch.from_numpy(move).long()

        return tensor, move"""



if __name__ == "__main__":
     data = import_data(end_idx=1)
     bitboards, moves = extract_bitboards(data)

     print(f"\n\n\n{len(ALL_MOVES)}\n\n\n")
     
     for game_idx in range(len(moves)):
         print(bitboards[game_idx])
         print(moves[game_idx])