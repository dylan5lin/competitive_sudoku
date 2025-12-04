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
        our_player = game_state.current_player
        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def possible(state, i, j, value):
            return state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in state.taboo_moves \
                       and (i, j) in state.player_squares()
        def evaluate_state(state:GameState):
            player = our_player
            opponent = 3 - player
            current_player = state.current_player
            board = state.board
            N = board.N
            # Score difference
            score_diff = state.scores[player - 1] - state.scores[opponent - 1]
            # Chances to complete next difference
            player_chances = chances_to_complete_next(state, player)
            opponent_chances = chances_to_complete_next(state, opponent)
            chance_diff = player_chances - opponent_chances
            # Regions completed difference
            m = state.board.region_height()
            n = state.board.region_width()
            for br in range(0, N, m):
                for bc in range(0, N, n):
                    player_regions = 0
                    opponent_regions = 0
                    empty = 0
                    for r in range(br, br + m):
                        for c in range(bc, bc + n):
                            cell_value = state.board.get((r, c))
                            if cell_value == SudokuBoard.empty:
                                empty += 1
                            elif (r, c) in state.occupied_squares1:
                                player_regions += 1
                            elif (r, c) in state.occupied_squares2:
                                opponent_regions += 1
            region_advantage = player_regions - opponent_regions
            center_control_diff = center_control(state, WEIGHTS, player)
            # TODO:             
            # Territorium control
            # Weights optimaliseren
            # Check what moves opponent can play next turn
            if current_player != player:
                opponent_legal_squares = len(state.player_squares())
            else:
                opponent_legal_squares = 0
            # Features
            features = [score_diff, chance_diff, center_control_diff, opponent_legal_squares, region_advantage]
            # Weights for features
            weights = [3, 5, 15, -7.5, 10]
            score = sum(f * w for f, w in zip(features, weights))
            return score
        # Ignore squares where player has 'full control' (only they can play there)
        def prioritize(state:GameState, player):
            N = state.board.N
            opponent = 3 - player
            prioritized_squares = []
            player_possible_moves = generate_all_moves(state)
            temp = state.current_player
            state.current_player = opponent
            opponent_possible_squares = generate_all_moves(state)
            opponent_squares = []
            for move in opponent_possible_squares:
                opponent_squares.append(move.square)
            state.current_player = temp
            for move in player_possible_moves:
                if move.square not in opponent_squares:
                    prioritized_squares.append(move)
            return prioritized_squares


            
                
        def center_weights(board: SudokuBoard): # Create weight map to assign higher weights in center squares.
            N = board.N
            center = (N-1)/2
            center_squares = (center, center)
            max_dist = (abs(0 - center_squares[0]) + abs(0 - center_squares[1]))
            weights = {}
            for row in range(N):
                for col in range(N):
                    dist = abs(row - center_squares[0]) + abs(col - center_squares[1])
                    weights[(row, col)] = max_dist - dist
            return weights
        
        WEIGHTS = center_weights(game_state.board)

        def center_control(state:GameState, weights, player): # Define center control depending on whether player dominates center squares.
            board = state.board
            N = board.N
            opponent = 3 - player
            player_score = 0
            opponent_score = 0
            player_squares = state.occupied_squares1 if player == 1 else state.occupied_squares2
            opponent_squares = state.occupied_squares2 if opponent == 2 else state.occupied_squares1
            # Calculate control scores based on weight map
            for (row,col), weight in weights.items():
                value = board.get((row, col))
                if value != SudokuBoard.empty:
                    if (row, col) in player_squares:
                        player_score += weight
                    elif (row, col) in opponent_squares:
                        opponent_score += weight

            return player_score - opponent_score # Final result = difference in center dominance.
        
        def chances_to_complete_next(state:GameState,curr_player):
            original = state.current_player
            state.current_player = curr_player 
            board = state.board
            #moves = generate_all_moves(state)
            legal_squares = state.player_squares()
            #legal_squares = {move.square for move in moves}
            N = board.N
            m, n = board.region_height(), board.region_width()
            if legal_squares is None:
                legal_squares = [(i,j) for i in range(N) for j in range(N)
                                 if board.get((i,j)) == SudokuBoard.empty]
            legal_squares = set(legal_squares)
            chances = 0
            
            # Check rows
            for i in range(N):
                empties = [(i, j) for j in range(N)
                        if board.get((i, j)) == SudokuBoard.empty]
                if len(empties) == 1 and empties[0] in legal_squares:
                    chances += 1

            # Check columns
            for j in range(N):
                empties = [(i, j) for i in range(N)
                        if board.get((i, j)) == SudokuBoard.empty]
                if len(empties) == 1 and empties[0] in legal_squares:
                    chances += 1
            # Check blocks
            for br in range(0, N, m):
                for bc in range(0, N, n):
                    empties = []
                    for i in range(br, br + m):
                        for j in range(bc, bc + n):
                            if board.get((i, j)) == SudokuBoard.empty:
                                empties.append((i, j))
                    if len(empties) == 1 and empties[0] in legal_squares:
                        chances += 1

            # Restore
            state.current_player = original
            return chances
        
        def sudoku_rules_satisfied(state, i,j,value):
            board = state.board
            # check row
            for col in range(N):
                if board.get((i, col)) == value:
                    return False
            #check column
            for row in range(N):
                if board.get((row, j)) == value:
                    return False
            # check block
            m  = board.region_height()
            n = board.region_width()
            block_row_start = (i // m) * m
            block_col_start = (j // n) * n
            for row in range(block_row_start, block_row_start + m):
                for col in range(block_col_start, block_col_start + n):
                    if board.get((row, col)) == value:
                        return False
            return True

        def generate_all_moves(state:GameState):
            moves = []
            for i in range(N):
                for j in range(N):
                    for value in range(1, N+1):
                        if possible(state, i, j, value) and sudoku_rules_satisfied(state, i,j,value):
                            moves.append(Move((i, j), value))
            moves.sort(key=lambda move: WEIGHTS[move.square], reverse=True)
            return moves


        def alpha_beta_pruning(state:GameState, depth, alpha, beta, is_maximizing):
            moves = generate_all_moves(state)
            prioritized_squares = prioritize(state, our_player)
            # Prioritize moves in prioritized squares
            moves.sort(key=lambda move: (move in prioritized_squares), reverse=True)
            moves = moves[:4]
            # Terminate search if no moves are available or depth = 0
            if moves == [] or depth == 0:
                return evaluate_state(state), None
            
            # Maximizing player
            if is_maximizing:
                max_eval = float('-inf')
                for move in moves:
                    #print(move.square)
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
            
            # Minimizing player
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
        
        def iterative_deepening_search(state:GameState, max_depth=5):
            init_alpha = float('-inf')
            init_beta = float('inf')
            current_best = None
            for depth in range(1, max_depth + 1):
                _, best_move = alpha_beta_pruning(state, depth, init_alpha, init_beta, True)
                if best_move is not None:
                    print(f"Depth: {depth}, Best Move: {best_move}, Value: {_}")
                    current_best = best_move
                    try:
                        self.propose_move(best_move)
                    except:
                        pass
            return current_best
                
        # Initial settings for iterative deepening
        MAX_DEPTH = 10
        all_moves = generate_all_moves(game_state)
        self.propose_move(random.choice(all_moves[:10]))
        best_move = iterative_deepening_search(game_state, MAX_DEPTH)
        if best_move is not None:
            self.propose_move(best_move)
        """
        while True:
            time.sleep(0.2) #?
            self.propose_move(random.choice(all_moves))
        """

