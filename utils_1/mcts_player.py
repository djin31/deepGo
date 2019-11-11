"""
Player using MCTS to play
"""

import numpy as np
from utils_1.goenv import GoEnv, create_env_copy
from utils_1.fnet import NeuralTrainer
import traceback
import time
import gc

class MCTSPlayer:
    def __init__ (self, board_size, fnet : NeuralTrainer, max_sims : int = 20):
        # Initialize the MonteCarlo class
        self.board_size = board_size
        self.num_actions = self.board_size ** 2 + 1
        self.fnet = fnet
        self.max_sims = max_sims

        np.set_printoptions(precision=3)

        # The state of the player
        self.state = GoEnv('black', self.board_size)
        self.state.reset()
        self.moves_played = 0

        # Hyperparameters
        self.cpuct = 1.5
        self.pass_invalid_thres = int(self.num_actions * 1 / 3) # Do not allow pass if >= these many positions are empty
        
        # Tracking the values
        self.Qsa = dict() # Stores Q values for s,a pairs
        self.Nsa = dict() # Stores the count for s,a pairs
        self.Ns = dict() # Count of number of times s is encountered

        self.Ps = dict() # Stores initial policy returned by the Fnet
        self.Ms = dict() # Stores list of valid moves
        self.Ts = dict() # Terminal states
        self.Vs = dict() # For storing values

    def play_opponent_move (self, opponent_action):
        """
        Play the opponent move
        """
        self.moves_played += 1
        try:
            state_copy = create_env_copy(self.state)
            state_copy.step(opponent_action)
            self.state.step(opponent_action)
        except:
            tb = traceback.format_exc()
            print (tb)
            print ("Opponent played an ILLEGAL MOVE!!")
            raise ValueError("MY OPPONENT PLAYED AN ILLEGAL MOVE")

    def play_game (self, TA_simulator):
        """
        Play one move by simulating
        """
        # Refresh the dictionaries every 300 moves
        if (np.random.rand() < 1 / 300):
            print ("Clearing the dicts @ Move #%d" % self.moves_played)
            self.clear_dicts()

        # Perform a simulation on the COPY of current state
        for _sim in range(self.max_sims):
            try:
                start_state = create_env_copy(self.state)
                self.run_simulator(start_state, depth=0)
            except:
                print ("BAD SIMULATION! EXCEPTION OCCURED")
            time.sleep(0.05 / self.max_sims) # Catch your breath

        action = self.play_move(TA_simulator)
        # policy = self._compute_pi(self.state)
        # self.play_move(policy[:])
        self.moves_played += 1

    def clear_dicts(self):
        """
        clear all dictionaries
        """
        self.Qsa = dict()
        self.Nsa = dict()
        self.Ns = dict()
        self.Ps = dict()
        self.Ms = dict()
        self.Ts = dict()
        gc.collect()
        time.sleep(0.1)

    def _get_legal_moves(self, state):
        """
        Get legal moves from this state
        If significant portion of the board is empty, and other moves are allowed, you SHOULD NOT pass
        """
        valid_moves = state.get_legal_moves()
        if (state.get_empty() > self.pass_invalid_thres and np.sum(valid_moves) > 0):
            valid_moves[-1] = 0 # No passing around!
        return valid_moves

    def play_move(self, TA_simulator):
        """
        Compute the policy and find best action
        """
        # Get the board representation of state
        s = self.state.hash_state()
        valid_moves = self._get_legal_moves(self.state)

        counts = np.array([self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.num_actions)])
        counts *= valid_moves # Masking with valid moves

        if np.sum(counts) == 0:
            print("All counts had to be masked :( !!")
            counts = valid_moves

        while np.sum(counts) > 0:
            # Choose a move and verify it first
            action = np.argmax(counts)

            try:
                # Verify with TA_simulator

                # Take action
                state_copy = create_env_copy(self.state)
                state_copy.step(action)
                self.state.step(action)

                # Return it
                return action
            except:
                # This action created some problem
                counts[action] = 0
                continue

        # No move is left
        print ('No action left :(')
        action = self.num_actions - 1 # Pass
        self.state.step(action)
        return action

    def run_simulator(self, state, depth, terminal_state=False):
        """
        Run one iteration of the MCTS simulation from the 'root': state
        Fig. 2 of paper
        a. Use the UCT to expand
        b. Once we hit the leaf node, use the FNet to compute values
        c. Update the path to the root with the value
        """
        # Get the board representation of state
        s = state.hash_state()
        stack = state.get_history()

        if s not in self.Ts:
            self.Ts[s] = state.isComplete()
            if self.Ts[s]:
                self.Ts[s] = -1 * state.player_turn() * state.get_winner()

        if self.Ts[s] == 1 or self.Ts[s] == -1:
            # This is a terminal state
            return -self.Ts[s]

        if terminal_state:
            # This is a terminal state not observed before
            val = -1 * state.player_turn() * state.get_winner()
            return -val

        if s not in self.Ps:
            # Leaf node
            p, v = self.fnet.predict(stack)
            valid_moves = self._get_legal_moves(state)
            p = p * valid_moves # masking invalid moves
            sum_p = np.sum(p)
            if sum_p > 0:
                p /= sum_p
            else:
                print ('All valid moves had to be masked!!')
                p = p + valid_moves
                if (np.sum(p) != 0):
                    p /= np.sum(p)
                else:
                    print ('NO VALID MOVE POSSIBLE !!!!!!!!!')
                    p = np.zeros(self.num_actions); p[self.num_actions - 1] = 1 # Pass
            
            self.Ms[s] = valid_moves
            self.Ps[s] = p
            self.Ns[s] = 0

            return -v

        if depth > 60:
            print ('hehe depth=', depth)
            if s in self.Vs:
                return -self.Vs[s]
            p, v = self.fnet.predict(stack)
            self.Vs[s] = v
            return -v

        # Pick the action with highest confidance bound
        def pick_action(s):
            valid_moves = self.Ms[s]
            best = -float('inf')
            best_action = -1

            def get_Q_plus_U(s, a):
                if (s,a) in self.Qsa:
                    return self.Qsa[(s,a)] + \
                            self.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
                else:
                    # Taking Q(s,a) = 0 here
                    return self.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s] + 1e-8)

            for a in range(self.num_actions):
                if valid_moves[a]:
                    # print (a, end=' ')
                    value = get_Q_plus_U(s, a)
                    if value > best:
                        best = value
                        best_action = a

            return best_action

        def play_next(state):
            s = state.hash_state()

            if (s[1] != state.player_turn()):
                print ('########################## OOOOOOOOOOOOO #########################')
                print (s[1])
                print (state.player_turn())
                print ('##################################################################')
                raise ValueError ("@Mankaran how is this possible?")

            try:
                a = pick_action(s)
                if a < 0 or a >= self.num_actions:
                    a = self.num_actions - 1 # pass
                _, _, done = state.step(a)
                return s, a, done
            except:
                self.Ms[s][a] = 0 # make this move invalid
                raise ValueError ('THE VALIDITY OF A MOVE CHANGED')

        # Play according to best action
        s, a, done = play_next(state)

        # Recursively call simulator on next state
        v = self.run_simulator(state, depth=depth+1, terminal_state=done)

        # Update Qsa
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)] * self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v
