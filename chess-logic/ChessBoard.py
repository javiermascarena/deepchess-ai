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

        #lists with all captured pieces
        self.white_captured_pieces = []
        self.black_captured_pieces = []

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
        """check if a move is possible, if it is it makes the move"""

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
        
        #if the piece to move is king or rook, castling wont be possible
        if isinstance(piece,(King,Rook)):
            piece.has_moved = True

        

        #end position piece
        end_pos_piece = self.board[end_pos[0]][end_pos[1]]

        #if there is no piece change the values
        if end_pos_piece == 0:
            self.board[end_pos[0]][end_pos[1]] = piece
            self.board[start_pos[0]][start_pos[1]] = 0
            piece.position = end_pos
        #the other piece is an enemy
        else:
            #check if the moving piece  is black or white
            if self.turn: #the piece is white
                #eliminate the piece
                self.white_captured_pieces.append(end_pos_piece)

                #modify the rest of values
                self.board[end_pos[0]][end_pos[1]] = piece
                self.board[start_pos[0]][start_pos[1]] = 0
                piece.position = end_pos

                #change turns
                self.turn = False

            else: #the piece is black
                #eliminate the piece
                self.black_captured_pieces.append(end_pos_piece)

                #modify the rest of values
                self.board[end_pos[0]][end_pos[1]] = piece
                self.board[start_pos[0]][start_pos[1]] = 0
                piece.position = end_pos

                #change turns
                self.turn = True






    def castling(self, king: King, rook: Rook):
        """checks if castling is allowed and if it is, it performs it"""

        # Check if they have already moved
        if king.has_moved or rook.has_moved:
            print("Castling is not possible")
            return
        
        # Check if there are pieces between them
        path = king.obtain_horizontal_path(king.position, rook.position)

        for square in path:
            if square != 0:
                print("There is a piece between both pieces, castling cant be done")
                return
            

        # When implemented add here if the king is under attack or possible check, castling wont be possible
        

        # Castling implementation
        if king.position[1] > rook.position[1]: # The queen is between them
            new_rook_pos = [rook.position[0], rook.position[1]+3]
            new_king_pos = [king.position[0], rook.position[1]-2]

        else: # No queen between them
            new_rook_pos = [rook.position[0], rook.position[1]-2]
            new_king_pos = [king.position[0], rook.position[1]+2]

        # Changes the positions for both pieces
        self.move_piece(king.position, new_king_pos)
        self.move_piece(rook.position, new_rook_pos)

        # Castling is only available once, change of has moved flag
        king.has_moved = True
        rook.has_moved = True

        

        




# Testing
if __name__ == "__main__":
    chess = ChessBoard() 
    for row in range(8):
        print(chess.board[row]) 