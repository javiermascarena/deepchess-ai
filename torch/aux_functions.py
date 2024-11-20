# Imports
import chess
import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time

# Whether to do the operations on the cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The layer number of each piece type in the tensor
PIECE_MAP = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
             chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

cumulative_times = {
    "import_data": 0,
    "board_to_tensor": 0,
    "possible_moves_to_tensor": 0,
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
    """Returns a list with the file paths of
      as many pgns as n_files"""

    # Getting the absolute path
    data_relative_path = os.path.join("..", "chess-data", "pgn")  # 2 ".." for the jupyter notebook
    data_absolute_path = os.path.abspath(data_relative_path)
    data = []

    iter = 0
    # Getting all the file paths in the data folder
    for file_name in os.listdir(data_absolute_path):
        file_path = os.path.join(data_absolute_path, file_name)

        if os.path.isfile(file_path):
            data.append(file_path)
            iter += 1

        # Returning the data when it has reached the file limit
        if iter == n_files: 
            return data

    # Returning all the pgns
    return data


@ timer_decorator
def board_to_tensor(board: chess.Board) -> np.array: 
    """Returns a 12x8x8 sparse tensor with ones where each piece is"""
    tensor = torch.zeros((14, 8, 8))

    # Searching each tile in the board
    for square in chess.SQUARES: 
        piece = board.piece_at(square)

        if piece:   
            # Obtain row and column of the piece in the board
            row, col = divmod(square, 8)  
            # Layer of the tensor the piece will be in
            piece_idx = PIECE_MAP[piece.piece_type]  
            # Different colors have different layers
            color = 0 if piece.color == chess.WHITE else 6  
            # Updating the tensor
            tensor[piece_idx + color, row, col] = 1

    tensor.to(device)
    
    return tensor


@ timer_decorator
def possible_moves_to_tensor(board: chess.Board, tensor: np.array) -> np.array: 
    """Returns the updated tensor with the 13th and 14th layers being the
       possible moves from black and white players"""
    
    # Extracting the legal moves of the position
    legal_moves = list(board.legal_moves)

    # Updating the tensor with all legal moves
    for move in legal_moves: 
        to_square = move.to_square
        to_row, to_col = divmod(to_square, 8)

        # When it's white turns, the legal moves of black (14th layer) 
        # is set to all zeros and viceversa
        if board.turn == chess.WHITE:
            tensor[12, to_row, to_col] = 1
        else: 
            tensor[13, to_row, to_col] = 1

    return tensor.to(device)


@ timer_decorator
def parse_pgn_to_tensors(data: list) -> list:
    """Gets a list of pgn file paths and returns the tensors and move positions
       for each of the games"""
    
    board_tensors = []
    next_moves = []

    # Looping through all files
    for pgn_file_path in data: 
        
        # Opening each file
        with open(pgn_file_path) as pgn_file:

            # Looping through all games
            while True: 
                # Reading the game
                game = chess.pgn.read_game(pgn_file) 
                
                # Stop when there are no more games
                if game is None: 
                    break

                # Getting the board
                board = game.board()

                # Getting the tensors for each move in the game
                for move in game.mainline_moves():

                    # Create tensor for the current state of the game
                    tensor = board_to_tensor(board)
                    tensor = possible_moves_to_tensor(board, tensor)
                    board_tensors.append(tensor)         

                    # Obtain the original position and the destination position after each move
                    from_square = move.from_square
                    to_square = move.to_square
           
                    # Update the board with the move
                    board.push(move)
                    
                    # Store this positions
                    next_moves.append((from_square, to_square))   

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
    pgns = import_data(10)

    # Convert pgns to tensors
    board_tensors, next_moves = parse_pgn_to_tensors(pgns)
    
    print(f"Total time for import_data: {cumulative_times['import_data']:.5f} seconds")
    print(f"Total time for board_to_tensor: {cumulative_times['board_to_tensor']:.5f} seconds")
    print(f"Total time for possible_moves_to_tensor: {cumulative_times['possible_moves_to_tensor']:.5f} seconds")