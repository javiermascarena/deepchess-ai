import numpy as np

class Piece: 

    def __init__(self, position: list[int, int], color: bool):
        """Initialization of variables"""

        # Stores the current position of the piece in the board
        self.position = position
        # True will be for white pieces and False for black pieces
        self.color = color
        # List that will store the possible moves of the piece
        self.possible_moves = []

    def check_inboard(self, position: list[int, int]) -> bool:
        """Checks whether a position is inside the board or not"""
        return 0 <= position[0] <= 7 and 0 <= position[1] <= 7


class Pawn(Piece):

    def __init__(self, position: list[int, int], color: bool):
        super().__init__(position, color)

    
    def possible_moves(self, board: list) -> None:
        """Calculates the possible moves the pawn can make"""

        # When the pawn is black it will move to a higher row in the board,
        # and when it is white to a lower one
        forward = -1 if self.color else 1

        # Checking the move to go 1 tile forward
        pos_check = [self.position[0], self.position[1] + forward]
   
        # Can only fo 1 tile forward if there is no other piece in that tile and 
        # it is not out of limits
        if board[pos_check[0], pos_check[1]] == 0 and self.check_inboard(pos_check):
            self.possible_moves.append(pos_check)

        # Checking the move to go 2 tiles forward
        pos_check = [self.position[0], self.position[1] + 2*forward]

        # Can only go 2 tiles forward if it is in the beginning position, and there 
        # is no other piece in that tile, and it is not out of limits
        if board[pos_check[0], pos_check[1]] == 0 and self.check_inboard(pos_check) and \
              ((self.position[0] == 6 and self.color) or (self.position[0] == 1 and not self.color)):
            self.possible_moves.append(pos_check)

        # Checking the moves to kill another piece diagonally
        pos_check = [[self.position[0] + 1, self.position[1] + forward], 
                     [self.position[0] - 1, self.position[1] + forward]]
        for pos in pos_check: 

            # Can only go diagonally if there is one piece there and it is not out limits
            if board[pos[0], pos[1]] != 0 and board[pos[0], pos[1]].color != self.color and \
                  self.check_inboard(pos):
                self.possible_moves.append(pos)


    def __repr__(self) -> str: 
        return "p" if self.color else "P"