import chess
import chess.pgn
import numpy as np
import os

# The layer number of each piece type in the tensor
PIECE_MAP = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
             chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def import_data(n_files=79) -> list: 
    """Returns a list with the file paths of
      as many pgns as n_files"""

    # Getting the absolute path
    data_relative_path = os.path.join("..", "data", "pgn")
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
    tensor = np.zeros((14, 8, 8))

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

    return tensor


def parse_pgn_to_tensors(data: list) -> list:
    """Gets a list of pgn file paths and returns the tensors
       for each of the games moves"""
    
    games_tensors = []

    # Looping through all files
    for pgn_file_path in data: 
        print(f"Processing file: {pgn_file_path}")  # Debugging output

        # Opening each file
        with open(pgn_file_path) as pgn_file:
            game_count = 0

            # Looping through all games
            while True: 
                # Reading the game
                game = chess.pgn.read_game(pgn_file) 
                
                # Stop when there are no more games
                if game is None: 
                    break

                # Getting the board
                board = game.board()
                moves_tensors = []

                # Getting the tensors for each move in the game
                for move in game.mainline_moves():

                    # Create tensor for the current state of the game
                    tensor = board_to_tensor(board)
                    tensor = possible_moves_to_tensor(board, tensor)
                    moves_tensors.append(tensor)              

                    # Update the board with the move
                    board.push(move)

                games_tensors.append(moves_tensors)
                
    return games_tensors