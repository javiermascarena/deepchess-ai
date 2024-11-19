# Imports
import chess
import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
from multiprocessing import Pool, cpu_count

# Whether to do the operations on the cpu or gpu
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

cumulative_times = {
    "import_data": 0,
    "parse_pgn_to_tensors": 0
}

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Add elapsed time to cumulative times
        func_name = func.__name__
        cumulative_times[func_name] += (end_time - start_time)

        return result
    
    return wrapper


@ timer_decorator
def import_data(n_files=79) -> list:
    """Returns a list of PGN file paths."""
    data_relative_path = os.path.join("..", "chess-data", "pgn")
    data_absolute_path = os.path.abspath(data_relative_path)
    data = []

    for i, file_name in enumerate(os.listdir(data_absolute_path)):
        if i >= n_files:
            break
        file_path = os.path.join(data_absolute_path, file_name)
        if os.path.isfile(file_path):
            data.append(file_path)

    return data


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Returns a 14x8x8 tensor for the board, with one layer per piece type."""
    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_idx = PIECE_MAP[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            tensor[piece_idx + color_offset, row, col] = 1

    return tensor


def update_legal_moves(board: chess.Board, tensor: torch.Tensor) -> None:
    """
    Efficiently updates the legal moves layers in the tensor.
    """
    # Determine which layer to update
    turn_layer = 12 if board.turn == chess.WHITE else 13
    
    # Clear legal move layers
    tensor[12:14] = 0

    # Extract all move destinations in a single operation
    legal_moves = list(board.legal_moves)
    move_indices = [move.to_square for move in legal_moves]

    # Convert move indices to rows and columns
    move_indices_tensor = torch.tensor(move_indices, dtype=torch.int64)
    rows = move_indices_tensor // 8  # Integer division to get row
    cols = move_indices_tensor % 8   # Modulus to get column

    # Update the appropriate layer for all moves
    tensor[turn_layer, rows, cols] = 1


@ timer_decorator
def parse_pgn_to_tensors(data: list) -> list:
    """Parses PGN files to generate tensors and move positions."""
    board_tensors = []
    next_moves = []

    for pgn_file_path in data:
        with open(pgn_file_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                board = game.board()
                tensor = board_to_tensor(board)  # Create initial tensor

                for move in game.mainline_moves():
                    update_legal_moves(board, tensor)
                    board_tensors.append(tensor.clone())  # Clone before update
                    next_moves.append((move.from_square, move.to_square))
                    board.push(move)  # Apply move to the board

    return board_tensors, next_moves


# Create a PyTorch dataset to store tensors and move positions
class ChessDataset(Dataset):
    def __init__(self, tensors, next_moves):
        """ Initializes the dataset with the tensors and the next move for each tensor"""
        
        self.data = []
        for i in range(len(next_moves)):
            self.data.append((tensors[i], next_moves[i]))

    def __len__(self):
        """ Returns the total number of moves in the dataset """
        return len(self.data)
    
    def __getitem__(self, idx):
        """ Returns a specific sample given by the index from the dataset: board tensor and move positions"""

        # Extracting the board tensor and move
        tensor, (from_pos, to_pos) = self.data[idx]

        # Converting the position of the moved piece to, to a pytorch tensor
        from_pos_tensor = torch.tensor(from_pos, dtype=torch.long)  
        # Converting the position to move the piece to, to a pytorch tensor
        to_pos_tensor = torch.tensor(to_pos, dtype=torch.long) 
        
        return tensor, from_pos_tensor, to_pos_tensor


if __name__ == "__main__":
    # Load pgn paths
    pgns = import_data(20)

    # Convert pgns to tensors
    board_tensors, next_moves = parse_pgn_to_tensors(pgns)
    
    print(f"Total time for import_data: {cumulative_times['import_data']:.5f} seconds")
    print(f"Total time for parse_pgn_to_tensors: {cumulative_times['parse_pgn_to_tensors']:.5f} seconds")