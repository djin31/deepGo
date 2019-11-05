"""
Class containing the player which interacts with Monte-Carlo to learn and play the game when reqd
"""

import numpy as np
import traceback
from montecarlo import MonteCarlo
from fnet import NeuralTrainer
import time

# from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Player:
    """
    Self-play bot, generating thousands of MCTS games, and training the neural network
    """
    def __init__ (self, board_size=13, mcts_sims=100, batch_size=10, fnet=None):
        """
        Initialize the Player class, instantiating the monte-carlo and Fnetwork
        """
        self.board_size = board_size
        self.mcts_sims = mcts_sims
        self.batch_size = batch_size # Update the network after generating these many number of games
        
        # Create the network
        self.fnet = NeuralTrainer(10, board_size)
        if fnet is not None:
            # Load the network from the file
            self.fnet.load_model(fnet)


    def self_play(self, number_games=10, checkpoint_path=None, logging=True, log_file=None):
        """
        Generate games from self-play and update the network
        """
        batch = []
        start_time = time.time()
        for game in range(number_games):
            # Generate a game
            print('\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print ('GAME %d' % game)
            print('#######################################################################3\n\n')
            try:
                simulator = MonteCarlo(self.board_size, self.fnet, self.mcts_sims) # Create a MCTS simulator
                game_batch = simulator.play_game()
                batch += game_batch

                if ((game + 1) % self.batch_size == 0): 
                    # Time to update the network
                    self.update_network(batch, checkpoint_path=checkpoint_path, logging=logging, log_file=log_file)
                    batch = [] # Empty the batch
            except:
                tb = traceback.format_exc()
            else:
                tb = "No error"
            finally:
                end_time = time.time()
                eprint("GAME # %d | Time Taken: %.3f secs" % (game, end_time - start_time))
                eprint(tb)
                start_time = end_time

        # Train for remaining in the batch
        if len(batch) > 0:
            # Time to update the network
            self.update_network(batch, checkpoint_path=checkpoint_path, logging=logging, log_file=log_file)
            batch = [] # Empty the batch

    def update_network(self, batch, checkpoint_path=None, logging=True, log_file=None):
        """
        Update the weights of the network
        """
        new_batch = self._augment_batch(batch)
        self.fnet.train(new_batch, logging=logging, log_file=log_file)

        # Save the network
        if checkpoint_path is not None:
            self.fnet.save_model(checkpoint_path)

    def _augment_batch(self, batch):
        """
        For each example in the batch, augment it with rotations and flips
        """
        new_batch = []
        for (s, pi, r) in batch:
            states = self._transform_state(s)
            policies = self._transform_policy(pi)

            for s_t, pi_t in zip(states, policies):
                new_batch.append((s_t, pi_t, r))

        return new_batch

    def _transform_state(self, stack):
        return self._produce_transformations(stack, threeD=True)

    def _transform_policy(self, policy):
        twoD_actions, pass_action = policy[0:-1].reshape(self.board_size, self.board_size), policy[-1] 

        transforms = self._produce_transformations(twoD_actions)
        return [
            np.array(list(t.flatten()) + [pass_action])
            for t in transforms
        ]

    def _produce_transformations(self, matrix, threeD=False):
        if threeD:
            axis = (1,2)
        else:
            axis = (0,1)
            
        matrices = []
        for times in range(4):
            mat = np.rot90(matrix, times, axis)
            flipped_mat = np.flip(mat, 1)
            matrices += [mat, flipped_mat]
        return matrices

if __name__ == '__main__':
    # Create a player
    player = Player(13, 200, 1)
    player.self_play(10, 'networks/testing2.model', logging=True, log_file='logs/log_testing2.txt')

