"""
Class containing the player which interacts with Monte-Carlo to learn and play the game when reqd
"""

import numpy as np
import traceback
from montecarlo import MonteCarlo
from fnet import NeuralTrainer
import time, os
from joblib import Parallel, delayed
import random
import pickle
from tqdm import tqdm


# from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# def generate_game_batch ():
#     """
#     A function independent of the big structures of the Player class
#     """


class Player:
    """
    Self-play bot, generating thousands of MCTS games, and training the neural network
    """
    def __init__ (self, board_size=13, mcts_sims=100, num_games=10, batch_size=100000, running_batch_file='running_batch.pkl', fnet=None,  load_running_batch=False):
        """
        Initialize the Player class, instantiating the monte-carlo and Fnetwork
        """
        self.board_size = board_size
        self.mcts_sims = mcts_sims
        self.num_games = num_games # Update the network after generating these many number of games
        self.batch_size = batch_size # Size of the running batch
        self.running_batch_file = running_batch_file # File for loading and saving the running file
        if load_running_batch:
            with open(self.running_batch_file, 'rb') as f:
                self.running_batch = pickle.load(f) # Running batch for training
        else:
            self.running_batch = [] # Running batch for training

        # Create the network
        self.fnet = NeuralTrainer(10, board_size, epochs=5)
        if fnet is not None:
            # Load the network from the file
            self.fnet.load_model(fnet)


    def self_play(self, total_games=10, checkpoint_path=None,  logging=True, log_file=None, game_offset=0):
        """
        Generate games from self-play and update the network
        """
        num_times = int(np.ceil(total_games / self.num_games))
        for g in range(num_times):
            game = g + game_offset
            # Catch your breath
            time.sleep(1)

            checkpoint_file = os.path.join(checkpoint_path, 'net%d.model' % game)

            # Generate a game
            print('\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print ('GAME %d' % game)
            print('#######################################################################3\n\n')

            def generate_batch ():
                game_batch = []
                try:
                    eprint ("instantiating sim")
                    simulator = MonteCarlo(self.board_size, self.fnet, self.mcts_sims) # Create a MCTS simulator
                    game_batch = simulator.play_game()
                except:
                    tb = traceback.format_exc()
                else:
                    tb = "No error"
                finally:
                    eprint(tb)
                    return game_batch

            start_time = time.time()
            games_batch = Parallel(n_jobs=self.num_games)(delayed(generate_batch)() for _i in range(self.num_games))
            batch = [b for gb in games_batch for b in gb]
            random.shuffle(batch)
            end_time = time.time()
            eprint("GAME # %d | Time Taken: %.3f secs" % ((game + 1) * self.num_games, end_time - start_time))

            # Update the network
            time.sleep(1) # Have some rest and collect some of your garbage in the meantime
            self.update_network(batch, checkpoint_path=checkpoint_file, logging=logging, log_file=log_file)

    def _chunks(self, chunk_size, num_chunks):
        """Yield successive n-sized chunks from l."""
        for _i in range(num_chunks):
            yield [random.choice(self.running_batch) for _c in range(chunk_size)] # With replacement
            # choices = np.random.choice(len(self.running_batch), chunk_size) # With replacement
            # yield self.running_batch[choices]

    def update_network(self, batch, checkpoint_path=None, logging=True, log_file=None):
        """
        Update the weights of the network
        """
        start_time = time.time()
        new_batch = self._augment_batch(batch)
        random.shuffle(new_batch)

        self.running_batch += new_batch
        self.running_batch = self.running_batch[-self.batch_size:]

        for chunk in tqdm(self._chunks(2048, int(2 * len(new_batch) / 2048))):
            self.fnet.train(chunk, logging=logging, log_file=log_file)

        if log_file is not None:
            with open(log_file, 'a') as lf:
                lf.write('\nTrained on running_batch of size %d/%d\n' % (2*len(new_batch), len(self.running_batch)))
                lf.write('---------------------------------------------------------------------------------------\n\n')

        # Save the network
        if checkpoint_path is not None:
            self.fnet.save_model(checkpoint_path)

        # Save the running batch
        with open(self.running_batch_file, 'wb') as f:
            pickle.dump(self.running_batch, f)

        eprint ('Network Updated. Time Taken: %d secs' % (time.time() - start_time))

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
    player = Player(13, 200, 10, running_batch_file='nov7/batch_file.pkl', fnet='nov7/prevnet.model', load_running_batch=True)
    player.self_play(500, 'nov7/', logging=True, log_file='nov7/training_log.txt', game_offset=3)
    # player = Player(13, 20, 10, running_batch_file='trash/batch_file.pkl')
    # player.self_play(20, 'trash/', logging=True, log_file='trash/training_log.txt')

