import time
import sys
from OthelloPosition import OthelloPosition
from AlphaBeta import AlphaBeta
from CountingEvaluator import CountingEvaluator

try:
    from ImprovedEvaluator import ImprovedEvaluator
except ImportError:
    ImprovedEvaluator = None


class Othello(object):
    """
    Main class for Othello that starts the game.
    Implements Iterative Deepening Search using the time limit argument.

    Author: Ola Ringdahl

    Args:
        arg1: Position to evaluate (a string of length 65 representing the board).
        arg2: Time limit in seconds
    """

    @staticmethod
    def iterative_deepening_search(position, time_limit):
        """
        Performs Iterative Deepening Search with Alpha-Beta pruning

        Args:
            position: The current game position
            time_limit: Time limit in seconds

        Returns:
            The best move found within the time limit
        """
        evaluator = ImprovedEvaluator() if ImprovedEvaluator is not None else CountingEvaluator()
        algorithm = AlphaBeta(evaluator)
        best_move = None
        depth = 1
        start_time = time.time()

        # Reserve time for overhead and final move selection.
        # Scale down the usable budget slightly for long limits to keep a safety margin.
        scaling_factor = 0.5 if time_limit < 8 else 0.45
        effective_time_limit = max(0.12, time_limit * scaling_factor)

        while time.time() - start_time < effective_time_limit:
            algorithm.set_search_depth(depth)
            remaining_time = effective_time_limit - (time.time() - start_time)

            # Stop if not enough time for next iteration
            if remaining_time < 0.08:
                break

            algorithm.set_time_limit(max(0.04, remaining_time * 0.85))
            algorithm.set_root_hint(best_move)

            try:
                current_move = algorithm.evaluate(position)

                # Only save move if depth was completed successfully
                if current_move is not None and not algorithm.time_up():
                    best_move = current_move
                    depth += 1
                else:
                    # Time ran out during this depth, use previous best
                    break

            except Exception:
                # If we encounter an error, use the best move so far
                break

        # If no move was found, return a pass move
        if best_move is None:
            moves = position.get_moves()
            if moves:
                best_move = moves[0]
            else:
                from OthelloAction import OthelloAction

                best_move = OthelloAction(0, 0, is_pass_move=True)

        return best_move


if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) < 3:
        print("Error: Too few arguments. Usage: python Othello.py <position_string> <time_limit>", file=sys.stderr)
        sys.exit(1)

    # Get command line arguments
    posString = sys.argv[1]
    if not posString:
        print("Error: Position string is missing.", file=sys.stderr)
        sys.exit(1)

    time_arg = sys.argv[2]
    if not time_arg:
        print("Error: Time limit argument is missing.", file=sys.stderr)
        sys.exit(1)

    try:
        time_limit = float(time_arg)
    except ValueError:
        print("Error: Time limit must be a number.", file=sys.stderr)
        sys.exit(1)

    # Validate input
    if len(posString) != 65:
        print(f"Error: Position string must be exactly 65 characters long (got {len(posString)})", file=sys.stderr)
        sys.exit(1)

    if posString[0] not in ["W", "B"]:
        print(f"Error: First character must be 'W' or 'B' (got '{posString[0]}')", file=sys.stderr)
        sys.exit(1)

    for i in range(1, 65):
        if posString[i] not in ["E", "O", "X"]:
            print(f"Error: Character at position {i} must be 'E', 'O', or 'X' (got '{posString[i]}')", file=sys.stderr)
            sys.exit(1)

    # Create position and find best move
    pos = OthelloPosition(posString)

    # Use Iterative Deepening Search
    move = Othello.iterative_deepening_search(pos, time_limit)

    # Print the move
    move.print_move()
