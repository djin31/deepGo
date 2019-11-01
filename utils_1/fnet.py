"""
Class for F-network for training the f_theta function
"""

import numpy as np

class FNet:
    def __init__ (self, board_size):
        # Initialize the network
        self.board_size = board_size
        self.pass_action = board_size * board_size
        self.total_actions = board_size * board_size + 1
        self.board_shape = (board_size, board_size)

        # ... other stuff

    def load_network(self, infile):
        # Load the network from a file
        with open(infile, 'rb') as f:
            print ('Loaded')

    def dump_network(self, outfile):
        # Dump the network in a file for future use
        with open(outfile, 'wb') as f:
            print ('Dumped')

    def foward_pass(self, state):
        # Doing forward pass through the FNet
        assert (state.shape == self.board_shape)

        value = np.random.rand() * 2 - 1
        policy = [np.random.rand() for i in range(0, self.total_actions)]
        policy = np.array(policy / sum(policy))

        return value, policy

    def update_net(self, batch):
        # Batch will be an array of (s, pi, r) tuples
        # s : State - 2D numpy array of shape self.board_size
        # pi : Policy values - 1D numpy array of size self.actions
        # r : +1, 0, -1 depending on whether white won

        pass
        