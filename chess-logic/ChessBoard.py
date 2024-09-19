import numpy as np
from Pieces import Piece,Pawn,Rook,Bishop,Queen,Knight,King


EMPTY_BOARD = [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]]


class ChessBoard:

    def __init__(self):
        """Initialize variables"""

        # The board will store a specific moment in the game
        # empty tiles will be filled with 0s
        self.board = EMPTY_BOARD

        # States which turn it is, T -> White, F -> False
        self.turn = True

        # Lists with all the pieces
        self.white_pieces = []
        self.black_pieces = []


    def initialize_game(self) -> None:
        """Initializes all pieces at the start of the program"""

        # Lists containing the pawns
        w_pawns = [Pawn([6, col], True) for col in range(8)]
        b_pawns = [Pawn([1, col], False) for col in range(8)]

        # Lists containing the rest of the pieces 
        w_pieces = [Rook([7,0], True), Knight([7,1],True), Bishop([7,2],True),Queen([7,3], True),
                    King([7,4],True), Bishop([7,5],True), Knight([7,6],True),Rook([7,7], True)]
        b_pieces = [Rook([0,0], False),Knight([0,1],False), Bishop([0,2],False),Queen([0,3], False),
                    King([0,4], False), Bishop([0,5], False), Knight([0,6], False), Rook([0,7], False)]

        # Lists containing all the pieces
        self.white_pieces = w_pawns + w_pieces
        self.black_pieces = b_pawns + b_pieces

    
    def set_board(self) -> None:
        """Sets the board before every game"""

        # Dividing the white pawns and rest of the pieces 
        w_pawns = [self.white_pieces[i] for i in range(8)]
        w_pieces = [self.white_pieces[i] for i in range(8, 16)]
        # Divideing the black pawns and the rest of the pieces
        b_pawns = [self.black_pieces[i] for i in range(8)]
        b_pieces = [self.black_pieces[i] for i in range(8, 16)]

        # Converting pawns back to the pawn class
        # if they promote in the previous game
        for i, pawn in enumerate(w_pawns):
            if not isinstance(pawn, Pawn): 
                w_pawns[i] = Pawn([0, 0], True)

            # No pawn is captured at the start
            w_pawns[i].captured = False

        for i, pawn in enumerate(b_pawns):
            if not isinstance(pawn, Pawn): 
                b_pawns[i] = Pawn([0, 0], False)

            # No pawn is captured at the start
            b_pawns[i].captured = False


        # No piece is captured at the start
        for piece in w_pieces: 
            piece.captured = False

        for piece in b_pieces: 
            piece.captured = False

        # Updating the pieces to the starting
        # positions
        for i in range(8):
            w_pawns[i].position = [6, i]
            w_pieces[i].position = [7, i]

            b_pawns[i].position = [1, i]
            b_pieces[i].position = [0, i]
        
        # Emptying the board
        self.board = EMPTY_BOARD
        
        # Showing the pieces on the board
        self.board[0] = b_pieces
        self.board[1] = b_pawns
        self.board[6] = w_pawns
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

        #check if they have already moved
        if king.has_moved or rook.has_moved:
            print("Castling is not possible")
            return
        
        #check if there are pieces between them
        path = self.obtain_path(king.position, rook.position)

        for square in path:
            if square != 0:
                print("There is a piece between both pieces, castling cant be done")
                return
            

        #when implemented add here if the king is under attack or possible check, castling wont be possible
        

        #castling implementation
        if king.position[1] > rook.position[1]: #the queen is between them
            new_rook_pos = [rook.position[0], rook.position[1]+3]
            new_king_pos = [king.position[0], rook.position[1]-2]

        else: #no queen between them
            new_rook_pos = [rook.position[0], rook.position[1]-2]
            new_king_pos = [king.position[0], rook.position[1]+2]

        #changes the positions for both pieces
        self.move_piece(king.position, new_king_pos)
        self.move_piece(rook.position, new_rook_pos)

        #castling is only available once, change of has moved flag
        king.has_moved = True
        rook.has_moved = True

        




    def obtain_path(self, pos1: list[int], pos2: list[int])-> list:
        """returns the squares between two squares,  the squares have to be in the same row"""
        
        #checks they are the same row
        if pos1[0] != pos2[0]:
            print("THey are not in the same row")
            return
        
        #initiate the path variable
        path = []


        if pos1[1]>pos2[1]: #position 1 is to the right of pos2

            #iterate through all squares
            for i in range(pos2[1]+1, pos1[1],1):
                path.append(self.board[pos1[0]][i]) #adds the square


        elif pos1[1] == pos2[1]: #the same position
            print("they are the same position")
            
        
        else:   #position 1 is to the left of pos2

            #iterate through all squares
            for i in range(pos1[1]+1, pos2[1], 1):
                path.append(self.board[pos1[0]][i]) #adds the square


        #return the list
        return path
    



        



        


        




# Testing
if __name__ == "__main__":
    chess = ChessBoard() 
    for row in range(8):
        print(chess.board[row]) 