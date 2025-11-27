#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from copy import deepcopy


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
        def possible(state, i, j, value):
            return state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in state.taboo_moves \
                       and (i, j) in state.player_squares()
        def evaluate_state(state:GameState):
            # Score difference
            score_diff = state.scores[0] - state.scores[1]
            # Filled squares difference
            squares_diff = len(state.occupied_squares1) - len(state.occupied_squares2)
            # Features
            features = [score_diff, squares_diff]
            # Weights for features
            weights = [5, 1]
            score = sum(f * w for f, w in zip(features, weights))
            return score
        
        def sudoku_rules_satisfied(state, i,j,value):

            # check row
            for col in range(N):
                if state.board.get((i, col)) == value:
                    return False
            #check column
            for row in range(N):
                if state.board.get((row, j)) == value:
                    return False
            # check block
            m  = state.board.region_height()
            n = state.board.region_width()
            block_row_start = (i // m) * m
            block_col_start = (j // n) * n
            for row in range(block_row_start, block_row_start + m):
                for col in range(block_col_start, block_col_start + n):
                    if state.board.get((row, col)) == value:
                        return False
            return True

        def generate_all_moves(state:GameState):
            moves = []
            for i in range(N):
                for j in range(N):
                    for value in range(1, N+1):
                        if possible(state, i, j, value) and sudoku_rules_satisfied(state, i,j,value):
                            moves.append(Move((i, j), value))
            return moves


        def alpha_beta_pruning(state:GameState, depth, alpha, beta, is_maximizing):
            if depth == 0:
                return evaluate_state(state), None
            moves = generate_all_moves(state)
            if moves == []:
                return evaluate_state(state), None
            if is_maximizing:
                max_eval = float('-inf')
                for move in moves:
                    child = deepcopy(state)
                    child.board.put(move.square, move.value)
                    eval, _ = alpha_beta_pruning(child, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                print("Max eval:", max_eval)
                return max_eval, best_move
            else:
                min_eval = float('inf')
                for move in moves:
                    child = deepcopy(state)
                    child.board.put(move.square, move.value)
                    eval, _ = alpha_beta_pruning(child, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                print("Min eval:", min_eval)
                return min_eval, best_move
        init_alpha = float('-inf')
        init_beta = float('inf')
        _, best_move = alpha_beta_pruning(game_state, 3, init_alpha, init_beta, True)
        if best_move is None:
            all_moves = generate_all_moves(game_state)
            self.propose_move(random.choice(all_moves))
        else:           
            self.propose_move(best_move)
        """
        while True:
            time.sleep(0.2) #?
            self.propose_move(random.choice(all_moves))
        """

