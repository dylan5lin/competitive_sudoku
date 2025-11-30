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
        def evaluate_board_1(state:GameState):
            # Simple evaluation function counting score difference, assuming we are player 1
            score = 0
            for i in range(N):
                for j in range(N):
                    cell_value = state.board.get((i, j))
                    if cell_value != SudokuBoard.empty:
                        if (i, j) in state.occupied_squares1:
                            score += 1
                        else:
                            score -= 1
            score = (score + state.scores[0]) * 5
            return score


        def evaluate_board_2(state:GameState):
            curr_player = state.current_player # index current player
            opp_player =  3- curr_player
            score_diff = state.scores[0] - state.scores[1]
            potential_next = chances_to_complete_next(state, 1) - chances_to_complete_next(state, 2)
            total_score = score_diff * 3 + potential_next * 10
            return total_score
        
        def chances_to_complete_next(state:GameState,curr_player):
            original = state.current_player
            state.current_player = curr_player 
            #moves = generate_all_moves(state)
            legal_squares = state.player_squares()
            #legal_squares = {move.square for move in moves}
            N = state.board.N
            m, n = state.board.region_height(), state.board.region_width()
            chances = 0
            for i in range(N):
                empty_row = []
                empty_col = []
                for j in range(N):
                    if state.board.get((i, j)) == SudokuBoard.empty:
                        empty_row.append((i, j))
                    if state.board.get((j, i)) == SudokuBoard.empty:
                        empty_col.append((j, i))
                if len(empty_row) == 1 and empty_row[0] in legal_squares:
                    chances += 1
                if len(empty_col) == 1 and empty_col[0] in legal_squares:
                    chances += 1
            
            # Check rows
            for i in range(N):
                empties = [(i, j) for j in range(N)
                        if state.board.get((i, j)) == SudokuBoard.empty]
                if len(empties) == 1 and empties[0] in legal_squares:
                    chances += 1

            # Check columns
            for j in range(N):
                empties = [(i, j) for i in range(N)
                        if state.board.get((i, j)) == SudokuBoard.empty]
                if len(empties) == 1 and empties[0] in legal_squares:
                    chances += 1
            # Check blocks
            for br in range(0, N, m):
                for bc in range(0, N, n):
                    empties = []
                    for i in range(br, br + m):
                        for j in range(bc, bc + n):
                            if state.board.get((i, j)) == SudokuBoard.empty:
                                empties.append((i, j))
                    if len(empties) == 1 and empties[0] in legal_squares:
                        chances += 1

            # Restore
            state.current_player = original
            return chances
        
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
                return evaluate_board_2(state), None
            moves = generate_all_moves(state)
            if moves == []:
                return evaluate_board_2(state), None
            if is_maximizing:
                max_eval = float('-inf')
                for move in moves:
                    child = deepcopy(state)
                    child.board.put(move.square, move.value)
                    child.current_player = 2 if state.current_player == 1 else 1

                    eval, _ = alpha_beta_pruning(child, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                
                return max_eval, best_move
            else:
                min_eval = float('inf')
                for move in moves:
                    child = deepcopy(state)
                    child.board.put(move.square, move.value)
                    child.current_player = 2 if state.current_player == 1 else 1

                    eval, _ = alpha_beta_pruning(child, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval, best_move
        init_alpha = float('-inf')
        init_beta = float('inf')
        _, best_move = alpha_beta_pruning(game_state, 2, init_alpha, init_beta, True)
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

