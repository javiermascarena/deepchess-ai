import numpy as np
from Pieces import Pawn

class ChessBoard:

    def __init__(self):
        """Initialize variables"""

        # The board will store a specific moment in the game
        # empty tiles will be filled with 0s
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],]

        # Chess pieces will be represented by the following scheme:
        # [pawn, rook, knight, bishop, queen, king] = [1, 2, 3, 4, 5, 6] 
        # for white pieces, and for black pieces, [7, 8, 9, 10, 11, 12]
        self.initialize_board()




    def initialize_board(self) -> None:
        """Fills the chess board at the start of the game"""

        # List containing the white pawns
        w_pawns = [Pawn([6, col], True) for col in range(8)]
        # Array containing the black pawns
        b_pawns = [Pawn([1, col], False) for col in range(8)]


        # Array containing white major pieces
        w_pieces = [2, 3, 4, 5, 6, 4, 3, 2]
        # Array containing black major pieces
        b_pieces = [8, 9, 10, 11, 12, 10, 9, 8]

        # Adding the pawns to the board 
        self.board[1] = b_pawns  
        self.board[6] = w_pawns  
        # Adding the major pieces to the board
        self.board[0] = b_pieces
        self.board[7] = w_pieces



# Testing
if __name__ == "__main__":
    chess = ChessBoard() 
    for row in range(8):
        print(chess.board[row]) 