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
import math
from tqdm import tqdm
import gc


# from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def generate_game_batch (board_size, fnet, mcts_sims, num_games):
    """
    A function independent of the big structures of the Player class
    """

    def play_game(board_size, fnet, mcts_sims):
        """
        Generate a single game and batch
        """
        game_batch = []; num_moves = -1
        try:
            eprint ("instantiating sim")
            simulator = MonteCarlo(board_size, fnet, mcts_sims) # Create a MCTS simulator
            game_batch, num_moves = simulator.play_game()
        except:
            tb = traceback.format_exc()
        else:
            tb = "No error"
        finally:
            eprint(tb)
            return game_batch, num_moves

    results = Parallel(n_jobs=num_games)(delayed(play_game)(board_size, fnet, mcts_sims) for i in range(num_games))
    games_batch = [r[0] for r in results]
    moves = [r[1] for r in results]
    eprint (moves)
    batch = [b for gb in games_batch for b in gb]
    gc.collect()

    return batch


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
        if not load_running_batch:
            # Put an empty batch inside the file
            with open(self.running_batch_file, 'wb') as f:
                pickle.dump([], f)

        # Create the network
        self.fnet = NeuralTrainer(10, board_size, epochs=1, batch_size=256, lr=0.05)
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

            start_time = time.time()
            batch = generate_game_batch(self.board_size, self.fnet, self.mcts_sims, self.num_games)
            random.shuffle(batch)
            end_time = time.time()
            eprint("GAME # %d | Time Taken: %.3f secs" % ((game + 1) * self.num_games, end_time - start_time))

            # Update the network
            gc.collect()
            time.sleep(1) # Have some rest and collect some of your garbage in the meantime
            self.update_network(batch, checkpoint_path=checkpoint_file, logging=logging, log_file=log_file)

    def _chunks(self, running_batch, chunk_size, num_chunks):
        """Yield successive n-sized chunks from l."""
        for _i in range(num_chunks):
            yield [random.choice(running_batch) for _c in range(chunk_size)] # With replacement
            # choices = np.random.choice(len(self.running_batch), chunk_size) # With replacement
            # yield self.running_batch[choices]

    def update_network(self, batch, checkpoint_path=None, logging=True, log_file=None):
        """
        Update the weights of the network
        """
        start_time = time.time()
        new_batch = self._augment_batch(batch)
        random.shuffle(new_batch)

        # Load and modify the running batch
        with open(self.running_batch_file, 'rb') as f:
            running_batch = pickle.load(f)

        running_batch += new_batch
        running_batch = running_batch[-self.batch_size:]

        for chunk in tqdm(self._chunks(running_batch, 16384, 5) ):
            self.fnet.train(chunk, logging=logging, log_file=log_file)

        # Save the network
        if checkpoint_path is not None:
            self.fnet.save_model(checkpoint_path)

        # Save the running batch
        with open(self.running_batch_file, 'wb') as f:
            pickle.dump(running_batch, f)

        if log_file is not None:
            with open(log_file, 'a') as lf:
                lf.write('\nTrained on running_batch of size %d/%d\n' % (16384 * 5, len(running_batch)))
                lf.write('---------------------------------------------------------------------------------------\n\n')

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
    player = Player(13, 200, 10, running_batch_file='nov10/batch_file.pkl', load_running_batch=True, fnet='nov10/net1.model')
    player.self_play(1000, 'nov10/', logging=True, log_file='nov10/training.txt', game_offset=2)
