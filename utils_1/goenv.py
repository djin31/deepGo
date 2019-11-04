import pachi_py
from copy import deepcopy
import numpy as np
import sys
import six
# from const import HISTORY, GOBAN_SIZE
HISTORY = 16

def _pass_action(board_size):
    return board_size ** 2


def _resign_action(board_size):
    return board_size ** 2 + 1


def _coord_to_action(board, c):
    """ Converts Pachi coordinates to actions """

    if c == pachi_py.PASS_COORD:
        return _pass_action(board.size)
    if c == pachi_py.RESIGN_COORD:
        return _resign_action(board.size)

    i, j = board.coord_to_ij(c)
    return i*board.size + j


def _action_to_coord(board, a):
    """ Converts actions to Pachi coordinates """

    if a == _pass_action(board.size):
        return pachi_py.PASS_COORD
    if a == _resign_action(board.size):
        return pachi_py.RESIGN_COORD

    return board.ij_to_coord(a // board.size, a % board.size)


def _format_state(history, player_color, board_size):
    """ 
    Format the encoded board into the state that is the input
    of the feature model, defined in the AlphaGo Zero paper 
    BLACK = 1
    WHITE = 2
    """
#     print(history.shape)
    to_play = np.full((1, board_size, board_size), player_color - 1)
    final_state = np.concatenate((history, to_play), axis=0)
    return final_state.astype(int)[:,:,:]
    


class GoEnv():

    def __init__(self, player_color, board_size):
        self.board_size = board_size
        self.history = np.zeros((HISTORY, board_size, board_size))
        self.steps = [-1,-2]

        colormap = {
            'black': pachi_py.BLACK,
            'white': pachi_py.WHITE,
        }
        self.player_color = colormap[player_color]

        self.komi = self._get_komi(board_size)
        self.state = _format_state(self.history,
                        self.player_color, self.board_size)
        self.done = True


    def _get_komi(self, board_size):
        """ Initialize a komi depending on the size of the board """

        if 14 <= board_size <= 19:
            return 7.5
        elif 9 <= board_size <= 13:
            return 7.5
        return 0
    

    def get_legal_moves(self):
        """ Get all the legal moves and transform their coords into 1d """

        legal_moves = self.board.get_legal_coords(self.player_color, filter_suicides=True)
        final_moves = np.zeros(self.board_size ** 2 + 1,dtype = int)

        for pachi_move in legal_moves:
            move = _coord_to_action(self.board, pachi_move)
            if move != self.board_size ** 2 or self.test_move(move):
                final_moves[move] = 1
        if(np.sum(final_moves)==0):
            final_moves[self.board_size**2] = 1 
        return final_moves


    def _act(self, action, history):
        """ Executes an action for the current player """

        self.board = self.board.play(_action_to_coord(self.board, action), self.player_color)
        self.steps.append(action)
        board = self.board.encode()
        color = self.player_color - 1
        a = int(HISTORY/2-1)
        for i in range(a):
            self.history[(a-i)*2+color] = self.history[(a-1-i)*2+color]  
#         self.history = np.roll(history, 1, axis=0)
        self.history[color] = np.array(board[color])
        self.player_color = pachi_py.stone_other(self.player_color)


    def test_move(self, action):
        """
        Test if a specific valid action should be played,
        depending on the current score. This is used to stop
        the agent from passing if it makes him loose
        """

        # board_clone = self.board.clone()
        current_score = self.board.fast_score  + self.komi

        # board_clone = board_clone.play(_action_to_coord(board_clone, action), self.player_color)
        # new_score = board_clone.fast_score + self.komi

        if self.player_color == 1 and current_score > 0 or self.player_color == 2 and current_score < 0:
           return False
        return True

    def give_Board_Copy(self):
        return self.board.clone()
    
    def player_turn(self):
        return 3-2*self.player_color
    
    def curr_score(self):
        return self.board.fast_score + self.komi
    
    def get_history(self):
        return _format_state(self.history, self.player_color, self.board_size)
    
    def isComplete(self):
        return (self.board.is_terminal or (self.steps[-1]==self.steps[-2] and self.steps[-1] == self.board_size**2))
    
    def stepsTaken(self):
        return self.steps[2:]
    
    def give_Board(self):
        board = self.board.encode()
        act_board = np.zeros(self.board_size**2,dtype = int).reshape(self.board_size,self.board_size)
        act_board[board[0]!=0] = 1
        act_board[board[1]!=0] = -1
        return act_board

    def hash_state(self):
        """
        Unrolled list version of board
        """
        return ' '.join( [str(e) for e in self.give_Board().flatten()] )

    def print_board(self):
        """
        Print the board
        X for Black, O for White, . for empty
        """
        board = self.give_Board()
        for row in board:
            for col in row:
                char = 'X' if col == 1 else ('O' if col == -1 else '.')
                print (char, end=' ')
            print()

    def reset(self):
        """ Reset the board """

        self.board = pachi_py.CreateBoard(self.board_size)
        opponent_resigned = False
        self.done = self.board.is_terminal or opponent_resigned
        return _format_state(self.history, self.player_color, self.board_size)


    def render(self):
        """ Print the board for human reading """

        outfile = sys.stdout
        outfile.write('To play: {}\n{}\n'.format(six.u(
                        pachi_py.color_to_str(self.player_color)),
                        self.board.__repr__().decode()))
        return outfile


    def get_winner(self):
        """ Get the winner, using the Tromp Taylor scoring + the komi """

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
    

    def step(self, action):
        """ Perfoms an action and choose the winner if the 2 player
            have passed """

        if not self.done:
            try:
                self._act(action, self.history)
            except pachi_py.IllegalMove:
                # print("alalalalala")
                self._act(self.board_size**2,self.history)
#                 six.reraise(*sys.exc_info())

        # Reward: if nonterminal, then the reward is -1
        if not self.board.is_terminal:
            if(self.steps[-1] == self.steps[-2] and self.steps[-1] == self.board_size**2):
                return _format_state(self.history, self.player_color, self.board_size), self.get_winner(), True
            else: 
                return _format_state(self.history, self.player_color, self.board_size), 0, False

        assert self.board.is_terminal
        self.done = True
        reward = self.get_winner()
        return _format_state(self.history, self.player_color, self.board_size), reward, True


    def __deepcopy__(self, memo):
        """ Used to overwrite the deepcopy implicit method since
            the board cannot be deepcopied """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "board":
                setattr(result, k, self.board.clone())
            else:
                setattr(result, k, deepcopy(v, memo))
        return result
    
def create_env_copy (Env):
    newEnv = GoEnv('black',Env.board_size)
    newEnv.reset()
    newEnv.history = deepcopy(Env.history)
    newEnv.player_color = deepcopy(Env.player_color)
    newEnv.state = _format_state(newEnv.history,newEnv.player_color, newEnv.board_size)
    newEnv.board = Env.give_Board_Copy()
    newEnv.done = deepcopy(Env.done)
    return newEnv