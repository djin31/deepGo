"""
Module containing the code for MonteCarlo class

Generates ONE complete game using the current FNetwork
Runs simulations and return a batch to update the neural network
"""

import numpy as np
from scipy.special import softmax
# from .game import Game
from .goenv import GoEnv, create_env_copy
from .fnet import NeuralTrainer

class MonteCarlo:
    def __init__ (self, board_size, fnet : NeuralTrainer, max_sims : int = 20, tau_thres : int = 30):
        # Initialize the MonteCarlo class
        self.board_size = board_size
        self.num_actions = self.board_size ** 2 + 1
        # self.game = Game(board_size)
        self.fnet = fnet
        self.max_sims = max_sims

        self.cpuct = 0.5 # Set arbitarily for now..

        # For how many moves to use the temperature parameter
        self.tau_thres = tau_thres

        # self.sims_done = 0

        # Initial state
        # self.state = game

        # Set of (s, pi, r) tuples
        # s here is the complete 13*13*17 state
        self.batch = []
        
        # Tracking the values
        self.Qsa = dict() # Stores Q values for s,a pairs
        self.Nsa = dict() # Stores the count for s,a pairs
        self.Ns = dict() # Count of number of times s is encountered

        self.Ps = dict() # Stores initial policy returned by the Fnet
        self.Ms = dict() # Stores list of valid moves
        self.Ts = dict() # Terminal states

    def play_game (self):
        """
        Play one full game, simulating on each move
        Returns the batch of (s, pi, r) tuples, for updating the fnet
        """
        # Initial state -- an instance of GoEnv
        self.state = GoEnv('black', self.board_size)
        self.state.reset()

        # while np.sum(self.state.legal_moves()) > 0:
        while not self.state.isComplete():
            # Perform a simulation on the COPY of current state
            for _sim in range(self.max_sims):
                start_state = create_env_copy(self.state)
                self.run_simulator(start_state)

            # Compute the policy from the root node and add to the batch
            # Add dummy reward to the batch for now, update at end of game
            policy = self._compute_pi(self.state)
            self.batch.append(self.state.get_history(), policy, 0)

            # Update state
            self.play_move(policy)

        # Update the reward and return the batch
        winner = -1 * self.state.get_winner()

        for idx, (s, pi, r) in enumerate(self.batch):
            player = s[16][0][0]
            r = winner * player
            self.batch[idx] = (s, pi, r)

        # Add the last terminal state to the batch
        policy = np.zeros(self.num_actions) # Do nothing
        self.batch.append(self.state.get_history(), policy, 1)

        return self.batch

    def _compute_pi(self, state):
        """
        Compute the policy as a softmax over N(s,a)
        """
        # Get the board representation of state
        s = state.give_Board()

        def get_count(s, a):
            return self.Nsa[(s,a)] if (s,a) in self.Nsa else 0

        counts = np.array(map(get_count, [(s,a) for a in range(self.num_actions)]))

        if self.tau_thres > 0:
            # Take action with softmax probabilities
            softmax_dist = softmax(counts)
            softmax_dist = softmax_dist * self.Ms[s] # Mask with valid moves
            softmax_dist /= np.sum(softmax_dist)

            self.tau_thres -= 1

            return softmax_dist

        else:
            # Tau is zero, take max action
            max_dist = np.zeros(self.num_actions)
            max_dist[np.argmax(counts)] = 1

            return max_dist



    def play_move(self, policy):
        """
        Choose an action according to the policy from the current state
        Execute and go to the next state
        """
        a = np.random.choice(np.arange(1, self.board_size ** 2 + 1), p=policy)
        self.state.step(a)

    def run_simulator(self, state):
        """
        Run one iteration of the MCTS simulation from the 'root': state
        Fig. 2 of paper
        a. Use the UCT to expand
        b. Once we hit the leaf node, use the FNet to compute values
        c. Update the path to the root with the value
        """
        # Get the board representation of state
        s = state.give_Board()
        stack = state.get_history()

        if s not in self.Ts:
            self.Ts[s] = state.isComplete()
            if self.Ts[s]:
                self.Ts[s] = -1 * state.player_turn() * state.get_winner()

        if self.Ts[s] is not False:
            # This is a terminal state
            return -self.Ts[s]

        if s not in self.Ps:
            # Leaf node
            p, v = self.fnet.predict(stack)
            valid_moves = state.get_legal_moves()
            p = p * valid_moves # masking invalid moves
            sum_p = np.sum(p)
            if sum_p > 0:
                p /= sum_p
            else:
                print ('All valid moves had to be masked!!')
                p = p + valid_moves
                p /= np.sum(p)
            
            self.Ms[s] = valid_moves
            self.Ps[s] = p
            self.Ns[s] = 0

            return -v

        # Pick the action with highest confidance bound
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

        for a in valid_moves:
            if a:
                value = get_Q_plus_U(s, a)
                if value > best:
                    best = value
                    best_action = a

        # Play according to best action
        a = best_action
        state.step(a)

        # Recursively call simulator on next state
        v = self.run_simulator(state)

        # Update Qsa
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)] * self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[(s,a)] += 1
        return -v



    # def update(self, state):
    #     self.states.append(state)
    #     pass

    # def get_play(self):
    #     while self.sims_done < self.max_sims:
    #         self.run_simulator()

