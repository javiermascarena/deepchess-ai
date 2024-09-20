import numpy as np
from Pieces import Piece, Pawn, Rook, Bishop, Queen, Knight, King


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
        self.board = np.array(EMPTY_BOARD, dtype=object)

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
                b_pawns[i] = Pawn(None, False)

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
        self.board = np.array(EMPTY_BOARD, dtype=object)
        
        # Showing the pieces on the board
        self.board[0] = b_pieces
        self.board[1] = b_pawns
        self.board[6] = w_pawns
        self.board[7] = w_pieces


    def move_piece(self, start_pos: list[int], end_pos: list[int]) -> None:
        """Makes a move checking if it is possible and updating
        the pieces and board afterward"""

        # The piece to be moved
        piece = self.board[start_pos[0], start_pos[1]]

        # Check if the element in the starting position exists
        if piece == 0:
            print("\nNo piece in this position")
            return

        # Check if its the correct turn        
        if piece.color != self.turn:
            print("\nIt's not your turn")
            return
        
        # Check possible moves
        piece.calculate_possible_moves(self.board)

        # Check if the final position is possible
        if end_pos not in piece.possible_moves:
            print(f"\n{piece}, to {end_pos} is not a possible move")
            return
        
        # Checks if the move is castling
        if isinstance(piece, King) and not piece.has_moved and\
            (end_pos == [piece.position[0], 2] or end_pos == [piece.position[0], 6]):
            # If the move is catling we perform it
            # First we obtain the rook to castle
            rook_pos = [piece.position[0], 0] if end_pos == [piece.position[0], 2]\
                else [piece.position[0], 7]
            rook = self.board[rook_pos[0], rook_pos[1]]

            # Updating the rook and king's positions
            rook.position = [piece.position[0], 3] if end_pos == [piece.position[0], 2]\
                else [piece.position[0], 5]
            piece.position = end_pos

            # Castling cannot be performed anymore
            piece.has_moved = True
            # Updating the turn
            self.turn = not self.turn 
            return  # Move has been performed

        # If the piece to move is king or rook, castling won't
        # be possible afterwards
        if isinstance(piece, (King, Rook)):
            piece.has_moved = True

        # End position piece
        end_pos_piece = self.board[end_pos[0], end_pos[1]]

        # If there is no piece change the values
        if end_pos_piece == 0:
            # Changing positions
            self.board[end_pos[0], end_pos[1]] = piece
            self.board[start_pos[0], start_pos[1]] = 0
            piece.position = end_pos

        # The other piece is an enemy piece
        else:   
            # Changing positions
            self.board[end_pos[0], end_pos[1]] = piece
            self.board[start_pos[0], start_pos[1]] = 0
            piece.position = end_pos
            # The piece is captured
            end_pos_piece.captured = True

        # Change turns
        self.turn = not self.turn


    def castling(self, king: King, rook: Rook) -> None:
        """Checks if castling is allowed and if it is added 
        to the king's possible moves"""

        # Check if they have already moved
        if king.has_moved or rook.has_moved:
            return
        
        # Whether it is the left rook trying to 
        # castle or the right one
        left_rook = True if rook.position[1] < king.position[1] else False

        # Checking if there is no pieces between the king and rook 
        if left_rook:  # For the castling to the left
            for i in range(1, 4):
                # If there is a piece between them castling 
                # is not possible
                if self.board[king.position[0], i] != 0: 
                    return 
                
        else:  # For castling to the right
            for i in range(5, 7):
                # If there is a piece between them castling 
                # is not possible
                if self.board[king.position[0], i] != 0:
                    return 

        # Obtaining the list of opponent pieces
        opposite_color = False if king.color else True
        if opposite_color: 
            opponent_pieces = self.white_pieces
        else: 
            opponent_pieces = self.black_pieces

        # Checking every possible movement of the opponent pieces to 
        # know if the king is under attack
        if king.position in [move for piece in opponent_pieces\
                             for move in piece.possible_moves]: 
            # The king is under attack -> Castling can't be performed
            return 

        # Adding the castling moves to the possible moves of both pieces
        if left_rook: # Castling to the left
            king.possible_moves.append([king.position[0], 2])

        else: # Castling to the right
            king.possible_moves.append([king.position[0], 6])


# When calculating the possible moves of pieces, the last two pieces should be the kings, 
# and castling should be computed separately from the king's possible moves, since castling
# needs the possible movements from all pieces (including the opponent's king) to know 
# if the king is under attack or not
        
# Have to implement en passant

# Testing
if __name__ == "__main__":
    chess = ChessBoard() 
    chess.initialize_game()
    chess.set_board()
    for row in range(8):
        print(chess.board[row]) 
