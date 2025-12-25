from OthelloEvaluator import OthelloEvaluator
from OthelloPosition import OthelloPosition


class ImprovedEvaluator(OthelloEvaluator):
    """
    An improved evaluator that considers multiple factors:
    - Piece count
    - Corner control
    - Edge control
    - Mobility (number of moves available)
    """

    def __init__(self):
        # Corner positions (most valuable)
        self.corners = [(1, 1), (1, 8), (8, 1), (8, 8)]

        # Positions adjacent to corners (dangerous early game)
        self.x_squares = [(2, 2), (2, 7), (7, 2), (7, 7)]
        self.c_squares = [(1, 2), (2, 1), (1, 7), (2, 8), (7, 1), (8, 2), (7, 8), (8, 7)]

        # Edge positions
        self.edges = []
        for i in range(1, 9):
            self.edges.extend([(1, i), (8, i), (i, 1), (i, 8)])

        # Remove corners from edges to avoid double counting
        for corner in self.corners:
            if corner in self.edges:
                self.edges.remove(corner)

    def evaluate(self, othello_position: OthelloPosition) -> float:
        """
        Evaluate the position considering multiple factors
        """
        white_score = 0
        black_score = 0

        # 1. Piece count (basic)
        white_pieces, black_pieces = self._count_pieces(othello_position)
        piece_diff = white_pieces - black_pieces

        # 2. Corner control (very important)
        corner_score = self._evaluate_corners(othello_position)

        # 3. Edge control
        edge_score = self._evaluate_edges(othello_position)

        # 4. Mobility (number of available moves)
        mobility_score = self._evaluate_mobility(othello_position)

        # 5. Positional weights (X-squares and C-squares)
        positional_score = self._evaluate_positions(othello_position)

        # Get total number of pieces to determine game phase
        total_pieces = white_pieces + black_pieces
        max_pieces = 64

        # Adjust weights based on game phase
        if total_pieces < 20:  # Early game
            piece_weight = 0.5
            corner_weight = 50
            edge_weight = 3
            mobility_weight = 10
        elif total_pieces < 50:  # Mid game
            piece_weight = 1
            corner_weight = 40
            edge_weight = 5
            mobility_weight = 5
        else:  # End game
            piece_weight = 10
            corner_weight = 25
            edge_weight = 3
            mobility_weight = 2

        # Combine scores with dynamic weights
        total_score = piece_diff * piece_weight + corner_score * corner_weight + edge_score * edge_weight + mobility_score * mobility_weight + positional_score

        return float(total_score)

    def _count_pieces(self, position):
        """Count white and black pieces"""
        white_count = 0
        black_count = 0

        for row in range(1, 9):
            for col in range(1, 9):
                if position.board[row][col] == "W":
                    white_count += 1
                elif position.board[row][col] == "B":
                    black_count += 1

        return white_count, black_count

    def _evaluate_corners(self, position):
        """Evaluate corner control"""
        score = 0
        for row, col in self.corners:
            if position.board[row][col] == "W":
                score += 1
            elif position.board[row][col] == "B":
                score -= 1
        return score

    def _evaluate_edges(self, position):
        """Evaluate edge control"""
        score = 0
        for row, col in self.edges:
            if position.board[row][col] == "W":
                score += 1
            elif position.board[row][col] == "B":
                score -= 1
        return score

    def _evaluate_mobility(self, position):
        """Evaluate mobility (number of moves available)"""
        # Count moves for current player
        current_moves = len(position.get_moves())

        # Count moves for opponent
        temp_pos = position.clone()
        temp_pos.maxPlayer = not temp_pos.maxPlayer
        opponent_moves = len(temp_pos.get_moves())

        # Return difference (positive favors white, negative favors black)
        if position.maxPlayer:  # White to move
            return current_moves - opponent_moves
        else:  # Black to move
            return opponent_moves - current_moves

    def _evaluate_positions(self, position):
        """Evaluate X-squares and C-squares (generally bad early game)"""
        score = 0

        # X-squares are bad (adjacent to corners diagonally)
        for row, col in self.x_squares:
            if position.board[row][col] == "W":
                score -= 2
            elif position.board[row][col] == "B":
                score += 2

        # C-squares are bad (adjacent to corners orthogonally)
        for row, col in self.c_squares:
            if position.board[row][col] == "W":
                score -= 1
            elif position.board[row][col] == "B":
                score += 1

        return score
