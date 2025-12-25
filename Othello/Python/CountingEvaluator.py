from OthelloEvaluator import OthelloEvaluator
from OthelloPosition import OthelloPosition

"""
  A simple evaluator that just counts the number of black and white squares 
  Author: Ola Ringdahl
"""


class CountingEvaluator(OthelloEvaluator):
    def evaluate(self, othello_position: OthelloPosition) -> float:
        black_squares = 0
        white_squares = 0
        for row in othello_position.board:
            for item in row:
                if item == "W":
                    white_squares += 1
                if item == "B":
                    black_squares += 1
        return float(white_squares - black_squares)
