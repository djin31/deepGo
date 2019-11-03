import pachi_py
import numpy as np
import sys
import six
from superGo import _coord_to_action
from superGo import _action_to_coord

class Board:
    def __init__(self,board,player_color):
        self.board = board
        self.player_color = player_color
        self.board_size = 13
        self.komi = 5.5
        
# def test_move(board,action,player_color):
#     new_score = board.fast_score  + komi
#     if player_color - 1 == 0 and new_score > 0 or player_color - 1 == 1 and new_score < 0:
#        return False
#     return True

    def get_board_moves(self):
        legal_moves = self.get_legal_coords(self.player_color, filter_suicides=True)
        final_moves = np.zeros(self.board_size ** 2 + 1,dtype = int)

        for pachi_move in legal_moves:
            move = _coord_to_action(self.board, pachi_move)
    #         if move != board_size ** 2 or test_move(board,move,player_color):
            final_moves[move] = 1
        return final_moves

    def npBoard(self):
        board = self.board.encode()
        act_board = np.zeros(169,dtype = int).reshape(13,13)
        act_board[board[0]!=0] = 1
        act_board[board[1]!=0] = -1
        return act_board

    def getWinner(self):
        score = self.board.fast_score + self.komi
        white_wins = score > 0
        black_wins = score < 0
        if(white_wins):
            reward = 1
        elif(black_wins):
            reward = -1
        else:
            reward = 0
        return reward

    def take_step(self,action):
        player_color = 1 if player == 1 else 2
        newBoard = board.play(_action_to_coord(board, action), player_color)
        return newBoard

    def copy_board(board):
        return board.clone()

    def score(board):
        return board.fast_score + komi

    def isCompleted(board):
        return board.is_terminal