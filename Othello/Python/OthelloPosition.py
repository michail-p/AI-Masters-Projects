import numpy as np
from OthelloAction import OthelloAction
from typing import List


class OthelloPosition(object):
    """
    This class is used to represent game positions. It uses a 2-dimensional char array for the board
    and a Boolean to keep track of which player has the move.

    For convenience, the array actually has two columns and two rows more that the actual game board.
    The 'middle' is used for the board. The first index is for rows, and the second for columns.
    This means that for a standard 8x8 game board, board[1][1] represents the upper left corner,
    board[1][8] the upper right corner, board[8][1] the lower left corner, and board[8][8] the lower left corner.

    Author: Ola Ringdahl
    """

    def __init__(self, board_str=""):
        """
        Creates a new position according to str. If str is not given all squares are set to E (empty)

        Args:
            board_str (str): A string of length 65 representing the board.
            board_str[0]  -> 'W' or 'B' (who moves)
            board_str[1:] -> 64 chars from {'E','O','X'} (E=Empty, O=White, X=Black)
        """
        self.BOARD_SIZE = 8
        self.maxPlayer = True  # True = White to move, False = Black to move
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        self.DIRS = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]

        # Use dtype='U1' (1-char Unicode) to keep memory small & fast.
        self.board = np.full((self.BOARD_SIZE + 2, self.BOARD_SIZE + 2), "E", dtype="U1")
        if len(board_str) >= 65:
            # Set player to move
            self.maxPlayer = board_str[0] == "W"
            for i in range(1, len(board_str)):
                col = ((i - 1) % 8) + 1
                row = (i - 1) // 8 + 1
                # For convenience we use W and B in the board instead of X and O:
                if board_str[i] == "X":
                    self.board[row][col] = "B"
                elif board_str[i] == "O":
                    self.board[row][col] = "W"

    def initialize(self):
        """
        Initializes the position by placing four coins in the middle of the board.
        """
        mid = self.BOARD_SIZE // 2
        self.board[mid][mid] = "W"
        self.board[mid + 1][mid + 1] = "W"
        self.board[mid][mid + 1] = "B"
        self.board[mid + 1][mid] = "B"
        self.maxPlayer = True

    def make_move(self, action: OthelloAction):
        """
        Perform the move suggested by the OhelloAction action and return the new position.
        This also changes the player to move next.

        Args:
            action (OthelloAction): The move to make as an OthelloAction

        Returns:
            The OthelloPosition resulting from making the move action in the current position.
        """
        new_pos = self.clone()

        if action.is_pass_move:
            new_pos.maxPlayer = not new_pos.maxPlayer
            return new_pos

        row, col = action.row, action.col

        if new_pos.maxPlayer:
            new_pos.board[row][col] = "W"
        else:
            new_pos.board[row][col] = "B"

        for dr, dc in self.DIRS:
            if self.__captures_in_direction(row, col, dr, dc):
                new_pos._flip_discs_in_direction(row, col, dr, dc)

        new_pos.maxPlayer = not new_pos.maxPlayer
        return new_pos

    def get_moves(self) -> List[OthelloAction]:
        """
        Get all possible moves for the current position

        Returns:
            A list of OthelloAction representing all possible moves in the position. If the list is empty, there are no legal moves for the player who has the move.
        """
        moves = []
        append = moves.append
        for r in range(1, self.BOARD_SIZE + 1):
            for c in range(1, self.BOARD_SIZE + 1):
                if self.__is_candidate(r, c) and self.__is_move(r, c):
                    append(OthelloAction(r, c))
        return moves

    # ------------ Private methods (helpers) ------------
    def _flip_discs_in_direction(self, row: int, col: int, dr: int, dc: int):
        """
        Flip all opponent discs in the given direction starting from (row, col)

        Args:
            row (int): Starting row position
            col (int): Starting column position
            dr (int): Delta row
            dc (int): Delta column
        """
        r = row + dr
        c = col + dc

        while 1 <= r <= self.BOARD_SIZE and 1 <= c <= self.BOARD_SIZE:
            if self.__is_opp_coin(r, c):
                if self.maxPlayer:
                    self.board[r][c] = "W"
                else:
                    self.board[r][c] = "B"
            else:
                break
            r += dr
            c += dc

    def __captures_in_direction(self, row: int, col: int, dr: int, dc: int) -> bool:
        """
        Does placing at (row, col) capture at least one opponent disc in (dr, dc)?
        Must see at least one opponent coin immediately, followed by own coin before hitting empty/out.

        Args:
            row (int): The row of the board position
            col (int): The column of the board position
            dr (int): Delta row
            dc (int): Delta column

        Returns:
            True if it flips at least one opponent disc in the direction (dr,dc)
        """
        size = self.BOARD_SIZE
        r = row + dr
        c = col + dc
        # First neighbor must be opponent
        if not self.__is_opp_coin(r, c):
            return False
        # Then at least one opp followed by an own
        r += dr
        c += dc
        # Walk until we find own or empty/border (no capture)
        while 1 <= r <= size and 1 <= c <= size:
            if self.board[r, c] == "E":
                return False
            if self.__is_own_coin(r, c):
                return True
            r += dr
            c += dc
        return False

    def __is_candidate(self, row: int, col: int) -> bool:
        """
        Check if a position is a candidate for a move (empty and has a neighbour)

        Args:
            row (int): The row of the board position
            col (int): The column of the board position

        Returns:
             True if it is a candidate
        """
        return self.board[row][col] == "E" and self.__has_neighbour(row, col)

    def __is_move(self, row: int, col: int) -> bool:
        """
        Check if it is possible to do a move from this position (can capture in at least one direction).

        Args:
            row (int): The row of the board position
            col (int): The column of the board position

        Returns:
             True if it is possible to do a move
        """
        if row < 1 or row > self.BOARD_SIZE or col < 1 or col > self.BOARD_SIZE:
            return False
        # Try all 8 directions. Return as soon as one captures
        for dr, dc in self.DIRS:
            if self.__captures_in_direction(row, col, dr, dc):
                return True
        return False

    def __is_opp_coin(self, row: int, col: int) -> bool:
        """
        Check if the position is occupied by the opponent

        Args:
            row (int): The row of the board position
            col (int): The column of the board position

        Returns:
            True if opponent coin
        """
        if self.maxPlayer and self.board[row][col] == "B":
            return True
        if not self.maxPlayer and self.board[row][col] == "W":
            return True
        return False

    def __is_own_coin(self, row: int, col: int) -> bool:
        """
        Check if the position is occupied by the player

        Args:
            row (int): The row of the board position
            col (int): The column of the board position

        Returns:
            True if own coin
        """
        if not self.maxPlayer and self.board[row][col] == "B":
            return True
        if self.maxPlayer and self.board[row][col] == "W":
            return True
        return False

    def __has_neighbour(self, row: int, col: int) -> bool:
        """
        Check if the position has any non-empty squares

        Args:
            row (int): The row of the board position
            col (int): The column of the board position

        Returns:
            True if it has neighbours
        """
        if self.board[row, col] != "E":
            return False
        b = self.board
        # Check the 8 surrounding squares
        for dr, dc in self.DIRS:
            if b[row + dr, col + dc] != "E":
                return True
        return False

    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state (no moves for either player)

        Returns:
            True if the game is over
        """
        if len(self.get_moves()) > 0:
            return False

        temp_pos = self.clone()
        temp_pos.maxPlayer = not temp_pos.maxPlayer
        if len(temp_pos.get_moves()) > 0:
            return False

        return True

    # ------------ Utility / Introspection ------------
    def to_move(self):
        """
        Check which player's turn it is

        Returns:
            True if the first player (white) has the move, otherwise False
        """
        return self.maxPlayer

    def clone(self):
        """
        Deep copy the current position

        Returns:
            A new OthelloPosition, identical to the current one.
        """
        # Create an uninitialized instance of the same class
        ot = type(self).__new__(type(self))
        # ot = MyOthelloPosition("")

        # Copy the simple fields
        ot.BOARD_SIZE = self.BOARD_SIZE
        ot.maxPlayer = self.maxPlayer
        ot.DIRS = self.DIRS

        # Deep copy the board
        ot.board = self.board.copy()

        return ot

    def print_board(self):
        """
        Prints the current board. Do not use when running othellostart (it will crash)
        """
        print(self.board)
        # print("ToMove: ", self.maxPlayer)
