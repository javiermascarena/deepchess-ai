import numpy as np

class Piece: 

    def __init__(self, position: list[int], color: bool):
        """Initialization of variables"""

        # Stores the current position of the piece in the board
        self.position = position
        # True will be for white pieces and False for black pieces
        self.color = color
        # List that will store the possible moves of the piece
        self.possible_moves = []
        # Wether the piece has been captured or not
        self.captured = False


    def check_inboard(self, position: list[int]) -> bool:
        """Checks whether a position is inside the board or not"""
        return 0 <= position[0] <= 7 and 0 <= position[1] <= 7
    

    def delete_outboard(self, positions: list) -> list: 
        """Returns the position of a list of positions that are in the board"""
        inboard_positions = []

        # Checking if each position is inboard
        for pos in positions: 
            if self.check_inboard(pos):
                inboard_positions.append(pos)

        return inboard_positions
    
    def reset_posible_moves(self) -> None:
        """Resets possible moves of a piece, which will be done each turn"""
        self.possible_moves = []



class Pawn(Piece):
    


    def calculate_possible_moves(self, board: list) -> None:
        """Calculates the possible moves the pawn can make"""

        # When the pawn is black it will move to a higher row in the board,
        # and when it is white to a lower one
        forward = -1 if self.color else 1
        
        # Checking the move to go 1 tile forward
        pos_check = [self.position[0], self.position[1] + forward]
        piece = board[pos_check[0]][pos_check[1]]

        # Can only fo 1 tile forward if there is no other piece in that tile and 
        # it is not out of limits
        if piece == 0 and self.check_inboard(pos_check):
            self.possible_moves.append(pos_check)

        # Checking the move to go 2 tiles forward
        pos_check = [self.position[0], self.position[1] + 2*forward]

        piece = board[pos_check[0]][pos_check[1]]
        # Can only go 2 tiles forward if it is in the beginning position, and there 
        # is no other piece in that tile, and it is not out of limits
        if piece == 0 and self.check_inboard(pos_check) and \
              ((self.position[0] == 6 and self.color) or (self.position[0] == 1 and not self.color)):
            self.possible_moves.append(pos_check)

        # Checking the moves to kill another piece diagonally
        pos_check = [[self.position[0] + 1, self.position[1] + forward], 
                     [self.position[0] - 1, self.position[1] + forward]]
        for pos in pos_check: 
            piece = board[pos[0]][pos[1]]

            # Can only go diagonally if there is one piece there and it is not out limits
            if piece != 0 and piece.color != self.color and \
                  self.check_inboard(pos):
                self.possible_moves.append(pos)


    def __repr__(self) -> str: 
        return "p" if self.color else "P"
    


class Rook(Piece):
    def __init__(self, position: list[int], color: bool):
        super().__init__(position, color)

        #variable to check if the rook has moved, for castling
        self.has_moved = False

    def calculate_possible_moves(self, board: list) -> None:
        """Calculates the possible moves the rook can make"""

        pos_check = []

        # All possible rook moves
        for i in range(8):
            pos_check.append([self.position[0], self.position[1] + i])
            pos_check.append([self.position[0], self.position[1] - i])
            pos_check.append([self.position[0] + i, self.position[1]])
            pos_check.append([self.position[0] - i, self.position[1]])

        # Removing outboard moves
        pos_check = self.delete_outboard(pos_check)

         #Checking every position
        for pos in pos_check:
            piece = board[pos[0]][pos[1]]

            #The movement can be done if there is no piece in the board
            #or the piece is of the opposite color
            if piece == 0 or piece.color != self.color:
                self.possible_moves.append(pos)
            
            #the color is the same
            else:
                #nothing has to be changed
                continue


    def __repr__(self) -> str: 
        return "r" if self.color else "R"



class Bishop(Piece):

    def calculate_possible_moves(self, board: list) -> None:
        """Calculates the possible moves the bishop can make"""

        pos_check = []

        # All posible bishop moves
        for i in range(8):
            pos_check.append([self.position[0] + i, self.position[1] + i])
            pos_check.append([self.position[0] - i, self.position[1] + i])
            pos_check.append([self.position[0] + i, self.position[1] - i])
            pos_check.append([self.position[0] - i, self.position[1] - i])

        # Removing outboard moves
        pos_check = self.delete_outboard(pos_check)

        #Checking every position
        for pos in pos_check:
            piece = board[pos[0]][pos[1]]

            #The movement can be done if there is no piece in the board
            #or the piece is of the opposite color
            if piece == 0 or piece.color != self.color:
                self.possible_moves.append(pos)
            
            #the color is the same
            else:
                #nothing has to be changed
                continue



    def __repr__(self) -> str: 
        return "b" if self.color else "B"



class Knight(Piece):
    
    def calculate_possible_moves(self, board: list) -> None:
        """Calculates the possible moves the knight can make"""

        pos_check = []

        # All possible knight moves
        pos_check.append([self.position[0] + 2, self.position[1] + 1])
        pos_check.append([self.position[0] + 2, self.position[1] - 1])
        pos_check.append([self.position[0] - 2, self.position[1] + 1])
        pos_check.append([self.position[0] - 2, self.position[1] - 1])
        pos_check.append([self.position[0] + 1, self.position[1] + 2])
        pos_check.append([self.position[0] - 1, self.position[1] + 2])
        pos_check.append([self.position[0] + 1, self.position[1] - 2])
        pos_check.append([self.position[0] - 1, self.position[1] - 2])

        # Deleting the moves that are outside the board
        pos_check = self.delete_outboard(pos_check)

        # Checking every position 
        for pos in pos_check: 
            piece = board[pos[0]][pos[1]] 

            # The movement can be done if there is no piece 
            # in the tile or the piece is of the opposite color
            if piece == 0 or piece.color is not self.color: 
                self.possible_moves.append(pos)
            
            #the piece is the same color
            else:
                #continue the rest of the loop
                continue


    def __repr__(self) -> str:
        return "k" if self.color else "K"
    


class Queen(Piece):

    def calculate_possible_moves(self, board: list) -> None:
        """Calculates the possible moves the queen can make"""

        pos_check = []
        for i in range(8):
            # Rook type moves
            pos_check.append([self.position[0], self.position[1] + i])
            pos_check.append([self.position[0], self.position[1] - i])
            pos_check.append([self.position[0] + i, self.position[1]])
            pos_check.append([self.position[0] - i, self.position[1]])
            # Bishop type moves
            pos_check.append([self.position[0] + i, self.position[1] + i])
            pos_check.append([self.position[0] - i, self.position[1] + i])
            pos_check.append([self.position[0] + i, self.position[1] - i])
            pos_check.append([self.position[0] - i, self.position[1] - i])

        # Deleting outboard moves
        pos_check = self.delete_outboard(pos_check)

        # Checking every position 
        for pos in pos_check: 
            piece = board[pos[0]][pos[1]] 

            # The movement can be done if there is no piece 
            # in the tile or the piece is of the opposite color
            if piece == 0 or piece.color is not self.color: 
                self.possible_moves.append(pos)
            
            #the piece is the same color
            else:
                #continue the rest of the loop
                continue


    def __repr__(self) -> str: 
        return "r" if self.color else "R"



class King(Piece): 

    def __init__(self, position: list[int], color: bool):
        super().__init__(position, color)

        #variable to check if the king has moved, for castling
        self.has_moved = False



    def calculate_possible_moves(self, board: list) -> None:
        """Calculates the possible moves the king can make"""

        pos_check = []

        # All possible king moves
        pos_check.append([self.position[0], self.position[1] + 1])
        pos_check.append([self.position[0], self.position[1] - 1])
        pos_check.append([self.position[0] + 1, self.position[1]])
        pos_check.append([self.position[0] - 1, self.position[1]])
        pos_check.append([self.position[0] + 1, self.position[1] + 1])
        pos_check.append([self.position[0] + 1, self.position[1] - 1])
        pos_check.append([self.position[0] - 1, self.position[1] + 1])
        pos_check.append([self.position[0] - 1, self.position[1] - 1])

        # Deleting the moves that are outside the board
        pos_check = self.delete_outboard(pos_check)

        # Checking every position 
        for pos in pos_check: 
            piece = board[pos[0]][pos[1]] 

            # The movement can be done if there is no piece 
            # in the tile or the piece is of the opposite color
            if piece == 0 or piece.color is not self.color: 
                self.possible_moves.append(pos)
            
            #the piece is the same color
            else:
                #continue the rest of the loop
                continue

    
    def __repr__(self) -> str:
        return "a" if self.color else "A"