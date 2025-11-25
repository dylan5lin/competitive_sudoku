#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def possible(i, j, value):
            return game_state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in game_state.taboo_moves \
                       and (i, j) in game_state.player_squares()
        
        def evaluate_board():
            # Simple evaluation function counting score difference
            score = 0
            for i in range(N):
                for j in range(N):
                    cell_value = game_state.board.get((i, j))
                    if cell_value != SudokuBoard.empty:
                        if (i, j) in game_state.occupied_squares1():
                            score += 1
                        else:
                            score -= 1
            return score
        
        def sudoku_rules_satisfied(i,j,value):

            # check row
            for col in range(N):
                if game_state.board.get((i, col)) == value:
                    return False
            #check column
            for row in range(N):
                if game_state.board.get((row, j)) == value:
                    return False
            # check block
            m  = game_state.board.region_height()
            n = game_state.board.region_width()
            block_row_start = (i // m) * m
            block_col_start = (j // n) * n
            for row in range(block_row_start, block_row_start + m):
                for col in range(block_col_start, block_col_start + n):
                    if game_state.board.get((row, col)) == value:
                        return False
            return True

        def generate_all_moves():
            moves = []
            for i in range(N):
                for j in range(N):
                    for value in range(1, N+1):
                        if possible(i, j, value) and sudoku_rules_satisfied(i,j,value):
                            moves.append(Move((i, j), value))
            return moves

        all_moves = generate_all_moves()

        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))

