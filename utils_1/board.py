"""
Class representing the Board datastructure
Used by MonteCarlo
Modify according to the pachi simulator

State: board_size * board_size 2D numpy array
Action: 0-(board_size*board_size) (the last action denoting pass)

White: 1
Black: -1

Important: The player is always white, so after playing as one player,
the board is flipped so that the other player may behave as white

As a test simulation -- we define a new game objective:
You can put stones on any empty place on the board
Objective: to maximize max no of stones in a row or col
"""
import numpy as np


class Board:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.pass_action = board_size * board_size
        self.board_shape = (board_size, board_size)

    def start(self):
        # Return a representation of the starting state of the game
        return np.zeros(self.board_shape, dtype=int)

    def next_state(self, state, action):
        # Take the action on state and return new state
        # DOES NOT modify inplace
        assert (0 <= action <= self.pass_action)
        assert (state.shape == self.board_shape)

        if action == self.pass_action:
            return state * -1

        new_state = -1 * state # Returning the flipped board
        row, col = action // self.board_size, action % self.board_size
        assert (new_state[row][col] == 0)
        new_state[row][col] = -1 # Playing as white, -1 as already flipped
        
        # Return the flipped board
        return new_state

    def legal_plays(self, state):
        # Return list of all legal actions
        # For this basic implementation: returns all the positions with empty places
        assert (state.shape == self.board_shape)
        return np.where(state.flatten() == 0)[0]

    def winner(self, state):
        # Based on the given state, return:
        # +1 if white is the winner
        # -1 if black is the winner
        # 0 if it's a draw
        assert (state.shape == self.board_shape)

        def calc_score(board):
            whites = board == 1
            row_sum = whites.sum(axis=1)
            col_sum = whites.sum(axis=0)
            return max(
                max(row_sum),
                max(col_sum)
            )

        white_score = calc_score(state)
        black_score = calc_score(-1 * state)

        if white_score > black_score:
            return 1
        elif white_score < black_score:
            return -1
        else:
            return 0