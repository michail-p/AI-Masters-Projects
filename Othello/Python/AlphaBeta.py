from OthelloAlgorithm import OthelloAlgorithm
from CountingEvaluator import CountingEvaluator
from OthelloAction import OthelloAction
from OthelloPosition import OthelloPosition
from OthelloEvaluator import OthelloEvaluator
import math
import time
from typing import Dict, Optional, Tuple

try:
    from ImprovedEvaluator import ImprovedEvaluator
except ImportError:  # Fallback if the improved evaluator is unavailable
    ImprovedEvaluator = None


class AlphaBeta(OthelloAlgorithm):
    """
    This is where you implement the alpha-beta algorithm.
        See OthelloAlgorithm for details

    Author:
    """

    DefaultDepth = 5
    POSITION_WEIGHTS = (
        (120, -20, 20, 5, 5, 20, -20, 120),
        (-20, -40, -5, -5, -5, -5, -40, -20),
        (20, -5, 15, 3, 3, 15, -5, 20),
        (5, -5, 3, 3, 3, 3, -5, 5),
        (5, -5, 3, 3, 3, 3, -5, 5),
        (20, -5, 15, 3, 3, 15, -5, 20),
        (-20, -40, -5, -5, -5, -5, -40, -20),
        (120, -20, 20, 5, 5, 20, -20, 120),
    )

    def __init__(self, othello_evaluator: Optional[OthelloEvaluator] = None, depth: int = DefaultDepth):
        if othello_evaluator is not None:
            self.evaluator = othello_evaluator
        elif ImprovedEvaluator is not None:
            self.evaluator = ImprovedEvaluator()
        else:
            self.evaluator = CountingEvaluator()
        self.search_depth = depth
        self.time_limit: Optional[float] = None
        self.start_time: Optional[float] = None
        self.transposition_table: Dict[Tuple[bytes, bool, int], float] = {}
        self.root_move_hint: Optional[OthelloAction] = None
        self.history_scores: Dict[Tuple[int, int, bool], float] = {}

    def set_evaluator(self, othello_evaluator: OthelloEvaluator) -> None:
        self.evaluator = othello_evaluator

    def set_search_depth(self, depth: int) -> None:
        self.search_depth = depth

    def set_time_limit(self, time_limit: float) -> None:
        self.time_limit = time_limit
        self.start_time = time.time()

    def set_root_hint(self, move: Optional[OthelloAction]) -> None:
        """Provide a preferred root move ordering hint for iterative deepening."""
        self.root_move_hint = move

    def time_up(self) -> bool:
        if self.time_limit is None or self.start_time is None:
            return False
        return time.time() - self.start_time >= self.time_limit

    def evaluate(self, othello_position: OthelloPosition) -> OthelloAction:
        moves = othello_position.get_moves()

        if not moves:
            return OthelloAction(0, 0, is_pass_move=True)

        if len(moves) == 1:
            return moves[0]

        maximizing_player = othello_position.maxPlayer

        # Order moves and put the hint (principal variation) first if available
        moves = self._order_moves(moves, maximizing_player=maximizing_player)

        if self.root_move_hint is not None:
            for idx, move in enumerate(moves):
                if move == self.root_move_hint:
                    if idx != 0:
                        moves.insert(0, moves.pop(idx))
                    break

        alpha = -math.inf
        beta = math.inf

        if maximizing_player:
            best_move = moves[0]
            best_value = -math.inf

            for move in moves:
                if self.time_up():
                    break

                new_position = othello_position.make_move(move)
                value = self.minimax(new_position, self.search_depth - 1, alpha, beta, False)

                if value > best_value:
                    best_value = value
                    best_move = move
                    alpha = max(alpha, value)

            self._update_history(best_move, maximizing_player, self.search_depth)
            return best_move
        else:
            best_move = moves[0]
            best_value = math.inf

            for move in moves:
                if self.time_up():
                    break

                new_position = othello_position.make_move(move)
                value = self.minimax(new_position, self.search_depth - 1, alpha, beta, True)

                if value < best_value:
                    best_value = value
                    best_move = move
                    beta = min(beta, value)

            self._update_history(best_move, maximizing_player, self.search_depth)
            return best_move

    def _order_moves(self, moves, maximizing_player: bool):
        """Order moves using a static positional weight table."""

        def weight(move):
            base = self.POSITION_WEIGHTS[move.row - 1][move.col - 1]
            history = self.history_scores.get((move.row, move.col, maximizing_player), 0.0)
            if maximizing_player:
                return base + history
            return base - history

        return sorted(moves, key=weight, reverse=maximizing_player)

    def _update_history(self, move: Optional[OthelloAction], maximizing_player: bool, depth: int) -> None:
        if move is None or move.is_pass_move:
            return
        key = (move.row, move.col, maximizing_player)
        bonus = float(depth * depth)
        self.history_scores[key] = self.history_scores.get(key, 0.0) + bonus
        if len(self.history_scores) > 4096:
            for existing_key in list(self.history_scores.keys()):
                self.history_scores[existing_key] *= 0.5
                if abs(self.history_scores[existing_key]) < 1.0:
                    self.history_scores.pop(existing_key)

    def _store_transposition(self, key: Tuple[bytes, bool, int], value: float) -> None:
        if len(self.transposition_table) > 200000:
            # Simple aging strategy: clear when table grows large
            self.transposition_table.clear()
        self.transposition_table[key] = value

    def minimax(self, position: OthelloPosition, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        if self.time_up():
            eval_result = self.evaluator.evaluate(position)
            return float(eval_result) if eval_result is not None else 0.0

        key = (position.board.tobytes(), maximizing_player, depth)
        if key in self.transposition_table:
            return self.transposition_table[key]

        if depth == 0 or position.is_terminal():
            eval_result = self.evaluator.evaluate(position)
            score = float(eval_result) if eval_result is not None else 0.0
            self._store_transposition(key, score)
            return score

        moves = position.get_moves()

        if not moves:
            new_position = position.clone()
            new_position.maxPlayer = not new_position.maxPlayer
            return self.minimax(new_position, depth - 1, alpha, beta, not maximizing_player)

        moves = self._order_moves(moves, maximizing_player)

        if maximizing_player:
            max_eval = -math.inf
            evaluated_any = False
            for move in moves:
                if self.time_up():
                    break
                new_position = position.make_move(move)
                eval_score = self.minimax(new_position, depth - 1, alpha, beta, False)
                evaluated_any = True
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self._update_history(move, maximizing_player, depth)
                    break
            if evaluated_any:
                self._store_transposition(key, max_eval)
            return max_eval
        else:
            min_eval = math.inf
            evaluated_any = False
            for move in moves:
                if self.time_up():
                    break
                new_position = position.make_move(move)
                eval_score = self.minimax(new_position, depth - 1, alpha, beta, True)
                evaluated_any = True
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self._update_history(move, maximizing_player, depth)
                    break
            if evaluated_any:
                self._store_transposition(key, min_eval)
            return min_eval
