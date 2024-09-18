import numpy as np
from Pieces import Piece,Pawn,Rook,Bishop,Queen,Knight,King

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

        #States which turn it is, T -> White, F -> False
        self.turn = True

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

        
        w_pieces = [Rook([7,0], True), Knight([7,1],True), Bishop([7,2],True),Queen([7,3], True),
                    King([7,4],True), Bishop([7,5],True), Knight([7,6],True),Rook([7,7], True)]


        b_pieces = [Rook([0,0], False),Knight([0,1],False), Bishop([0,2],False),Queen([0,3], False),
                    King([0,4], False), Bishop([0,5], False), Knight([0,6], False), Rook([0,7], False)]



        # Array containing white major pieces
        #w_pieces = [2, 3, 4, 5, 6, 4, 3, 2]
        # Array containing black major pieces
        #b_pieces = [8, 9, 10, 11, 12, 10, 9, 8]

        # Adding the pawns to the board 
        self.board[1] = b_pawns  
        self.board[6] = w_pawns  
        # Adding the major pieces to the board
        self.board[0] = b_pieces
        self.board[7] = w_pieces

    def move_piece(self, start_pos: list[int,int], end_pos: list[int,int]):

        piece = self.board(start_pos[0], start_pos[1])

        #check if the element in the starting position exists
        if self.piece == 0:
            print("No piece in this position")
            return


        #check if its the correct turn        
        if piece.color != self.turn:
            print("It`s not your turn")
            return
        
        #check possible moves
        piece.calculate_possible_moves(self.board)

        #check if the final position is possible
        if end_pos not in piece.possible_moves:
            print("Not a possible move")
            return
        
        #change the values
        self.board[end_pos[0], end_pos[1]] = piece
        self.board[start_pos[0], start_pos[1]] = 0
        piece.position = end_pos

        #switch turns 
        if self.turn:
            self.turn == False
        else:
            self.turn == True




# Testing
if __name__ == "__main__":
    chess = ChessBoard() 
    for row in range(8):
        print(chess.board[row]) 