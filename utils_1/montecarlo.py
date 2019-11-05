"""
Module containing the code for MonteCarlo class

Generates ONE complete game using the current FNetwork
Runs simulations and return a batch to update the neural network
"""

import numpy as np
from scipy.special import softmax
# from .game import Game
from goenv import GoEnv, create_env_copy
from fnet import NeuralTrainer

class MonteCarlo:
    def __init__ (self, board_size, fnet : NeuralTrainer, max_sims : int = 20, tau_thres : int = 30):
        # Initialize the MonteCarlo class
        self.board_size = board_size
        self.num_actions = self.board_size ** 2 + 1
        self.fnet = fnet
        self.max_sims = max_sims

        np.set_printoptions(precision=3)

        # Hyperparameters
        self.cpuct = 1.5
        self.tau_thres = tau_thres # For how many moves to use the temperature parameter
        self.pass_invalid_thres = int(self.num_actions * 1 / 3) # Do not allow pass if >= these many positions are empty
        # self.moves_executed = 0 # How many moves have already been played

        # Set of (s, pi, r) tuples
        # s here is the complete 17*13*13 state
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
        root_state = True # Whether this is the first state

        move_no = 1
        while move_no <= 450 and not self.state.isComplete():
            # Print state
            self.state.print_board()
            print ('-----------------------------------------------------------------')
            # Perform a simulation on the COPY of current state
            for _sim in range(self.max_sims):
                # print ('*****************************************************************')
                # print ('Sim #%d' % _sim)
                start_state = create_env_copy(self.state)
                self.run_simulator(start_state)

            # Compute the policy from the root node and add to the batch
            # Add dummy reward to the batch for now, update at end of game
            policy = self._compute_pi(self.state)
            self.batch.append((self.state.get_history(), policy, 0))

            # Update state
            # a = self.state.get_legal_moves()
            # print(a[0:-1].reshape(self.board_size,self.board_size))
            # print(a[self.board_size**2])
            print("Move #%d" % move_no); move_no += 1
            self.play_move(policy[:], root_state=root_state)
            root_state = False

        # Update the reward and return the batch
        print(self.state.stepsTaken())
        winner = -1 * self.state.get_winner()
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        self.state.print_board()
        print ("And the winner is .... %s !" % ('White' if winner == -1 else 'Black'))

        for idx, (s, pi, r) in enumerate(self.batch):
            player = 1 if s[16][0][0] == 1 else -1
            r = winner * player
            self.batch[idx] = (s, pi, r)

        # Add the last terminal state to the batch
        policy = np.zeros(self.num_actions) # Do nothing
        self.batch.append((self.state.get_history(), policy, 1))

        return self.batch

    def _get_legal_moves(self, state):
        """
        Get legal moves from this state
        If significant portion of the board is empty, and other moves are allowed, you SHOULD NOT pass
        """
        valid_moves = state.get_legal_moves()
        if (state.get_empty() > self.pass_invalid_thres and np.sum(valid_moves) > 0):
            valid_moves[-1] = 0 # No passing around!
        return valid_moves

    def _compute_pi(self, state):
        """
        Compute the policy proportional N(s,a)
        """
        # Get the board representation of state
        s = state.hash_state()
        valid_moves = self._get_legal_moves(state)

        counts = np.array([self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.num_actions)])
        print ('counts:');print (counts[0:-1].reshape(self.board_size, self.board_size), counts[-1]); print (sum(counts))
        print('valid_moves:');print (valid_moves[0:-1].reshape(self.board_size, self.board_size), valid_moves[-1])
        counts *= valid_moves # Masking with valid moves

        if np.sum(counts) == 0:
            print("All counts had to be masked :( !!")
            counts = valid_moves

        if self.tau_thres > 0:
            # Take action with proportional probabilities
            self.tau_thres -= 1
            return counts / float(np.sum(counts))
        else:
            # Tau is zero, take max action
            max_dist = np.zeros(self.num_actions)
            max_dist[np.argmax(counts)] = 1.0
            return max_dist

    def play_move(self, policy, root_state):
        """
        Choose an action according to the policy from the current state
        Execute and go to the next state
        Add Dirichlet noise always if self.tau_thres > 0 (i.e. for first 30 moves)
        P(s, a) = (1 − e)pa + ena, where n ∼ Dir(0.03) and e = 0.25
        """
        if self.tau_thres > 0:
            noise = np.random.dirichlet(alpha=((0.3,)*self.num_actions))
            policy = 0.75 * policy + 0.25 * noise

            valid_moves = self._get_legal_moves(self.state)
            policy *= valid_moves; policy /= np.sum(policy)

        # print(policy[169],"jj")
        print(self.num_actions)
        a = np.random.choice(np.arange(0, self.num_actions), p=policy)
        self.state.step(a)
        print("Played %s" % a)

    def run_simulator(self, state, terminal_state=False):
        """
        Run one iteration of the MCTS simulation from the 'root': state
        Fig. 2 of paper
        a. Use the UCT to expand
        b. Once we hit the leaf node, use the FNet to compute values
        c. Update the path to the root with the value
        """
        np.set_printoptions(precision=5, suppress=True)
        # Get the board representation of state
        s = state.hash_state()
        stack = state.get_history()

        # print(s)

        # state.print_board()
        # print('.......................................................................')

        # print('============================================')
        # print ('IN MCTS')
        # state.print_board()
        # # print (s)
        # # if s in self.Ms:
        # #     print ('not a leaf node')
        # #     print (self.Ms[s])
        # print ('------------------------------------------')

        # print (s)
        # print (self.Ts)
        # print (self.Ps)
        # print (self.Ns)
        # print ('=======================================')

        if s not in self.Ts:
            self.Ts[s] = state.isComplete()
            if self.Ts[s]:
                self.Ts[s] = -1 * state.player_turn() * state.get_winner()

        if self.Ts[s] == 1 or self.Ts[s] == -1:
            # print ('hobaod')
            # print (s)
            # print (self.Ts[s], type(self.Ts[s]))
            # print (self.Ts[s] is not False)
            # a = np.bool(False)
            # print (False is not False)
            # print (a is not False)
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
                # p = 0
            
            self.Ms[s] = valid_moves
            self.Ps[s] = p
            self.Ns[s] = 0

            return -v

        # Pick the action with highest confidance bound
        # valid_moves = self._get_legal_moves(state)
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

        # p2, p1 = self.Ps[s][0:-1].reshape(self.board_size, self.board_size), self.Ps[s][-1]
        # print (p2); print(p1)

        # values = np.array([get_Q_plus_U(s,a) if valid_moves[a] else -1 for a in range(self.num_actions)])
        # val2, val = values[0:-1].reshape(self.board_size, self.board_size), values[-1]

        # print (val2); print(val)
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print()

        # Play according to best action
        a = best_action
        # print (valid_moves)
        # print ('Simulator: Going to play: %d' % a)
        _, _, done = state.step(a)

        # Recursively call simulator on next state
        v = self.run_simulator(state, terminal_state=done)

        # Update Qsa
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)] * self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v
