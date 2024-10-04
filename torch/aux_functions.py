import chess
import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

# Whether to do the operations on the cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The layer number of each piece type in the tensor
PIECE_MAP = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
             chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def import_data(n_files=79) -> list: 
    """Returns a list with the file paths of
      as many pgns as n_files"""

    # Getting the absolute path
    data_relative_path = os.path.join("..", "..", "chess-data", "pgn")
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


def possible_moves_to_tensor(board: chess.Board, tensor: np.array) -> np.array: 
    """Returns the updated tensor with the 13th and 14th layers being the
       possible moves from black and white players """
    
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
    
    tensor.to(device)

    return tensor


def parse_pgn_to_tensors(data: list) -> list:
    """Gets a list of pgn file paths and returns the tensors and move positions
       for each of the games"""
    
    games_tensors = []
    move_positions = []

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
                game_tensors = []
                game_move_positions = []

                # Getting the tensors for each move in the game
                for move in game.mainline_moves():

                    # Create tensor for the current state of the game
                    tensor = board_to_tensor(board)
                    tensor = possible_moves_to_tensor(board, tensor)
                    game_tensors.append(tensor)         

                    # Obtain the original position and the destination position after each move
                    from_square = move.from_square
                    to_square = move.to_square

                    # Converting the format to row, col
                    from_pos = divmod(from_square, 8)
                    to_pos = divmod(to_square, 8)     

                    # Store this positions
                    game_move_positions.append((from_pos,to_pos))


                    # Update the board with the move
                    board.push(move)

                # Added the tensors and the moves for each game
                games_tensors.append(game_tensors)
                move_positions.append(game_move_positions)
                
    return games_tensors, move_positions




# Create a PyTorch dataset to store tensors and move positions
class ChessDataset(Dataset):
    def __init__(self, games_tensors, move_positions):
        """ Initializes the dataset  with the game tensors and the different moves
            games_tensors: list of lists, each containing the tensor for each game
            move_positions: list of lists, each containing the move (from_pos, to_pos)"""
        
        self.data = []

        # Join both lists together to have the dataset
        # Loop through both lists to do this
        for games_idx in range(len(games_tensors)):
            # Obtain tensors for the current game
            game_tensors = games_tensors[games_idx]

            # Obtain the moves (starting and end positions) for the current game
            game_moves = move_positions[games_idx]

            # Loop through each move in the current game
            for move_idx in range(len(game_tensors)):
                # Obtain the board tensor for the current move
                tensor = game_tensors[move_idx]

                # Obtain the current positions (starting and end positions) for the current move
                move = game_moves[move_idx]

                # Add both the tensor and move to the data
                self.data.append((tensor,move))

    def __len__(self):
        """ Returns the total number of moves in the dataset """
        return len(self.data)
    
    def __getitem__(self, idx):
        """ Returns a specific sample given by the index from the dataset: board tensor and move positions"""
        tensor, (from_pos, to_pos) = self.data[idx]

        # Convert move positions to PyTorch tensors 
        # In neural networks and training this will be necessary and better for pytorch
        # They have to be of type long for the loss calculation
        from_pos_tensor = torch.tensor(from_pos, dtype=torch.long)  
        to_pos_tensor = torch.tensor(to_pos, dtype=torch.long) 

        # Return the board state and move
        # A neural network usually works with float number which is why the board tensor is of type float
        return torch.tensor(tensor, dtype=torch.float32), from_pos_tensor, to_pos_tensor




"""# Code for testing chess dataset
# Load PGN files
pgn_files = import_data(n_files=1)

#  Parse PGN files to extract tensors and move positions
games_tensors, move_positions = parse_pgn_to_tensors(pgn_files)

#  Create a PyTorch dataset with the extracted data
chess_dataset = ChessDataset(games_tensors, move_positions)

#  Use DataLoader for batching and shuffling
# Batch Size 32 defines how many sample will be processed together
# Shuffle radomizes the order of samples in the dataset
dataloader = DataLoader(chess_dataset, batch_size=32, shuffle=True)

# Example: Loop through the DataLoader
for board_tensor, start_pos, end_pos in dataloader:
    print(f"Board Tensor Shape: {board_tensor.shape}")
    print(f"Start Position: {start_pos}")
    print(f"End Position: {end_pos}")"""