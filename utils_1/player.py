"""
Class containing the player which interacts with Monte-Carlo to learn and play the game when reqd
"""

import numpy as np
from .montecarlo import MonteCarlo
from .fnet import NeuralTrainer

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
        for game in range(number_games):
            # Generate a game
            simulator = MonteCarlo(self.board_size, self.fnet, self.mcts_sims) # Create a MCTS simulator
            game_batch = simulator.play_game()
            batch += game_batch

            if (game % number_games == 0): 
                # Time to update the network
                self.fnet.train(batch, logging=logging, log_file=log_file)
                batch = [] # Empty the batch

                # Save the network
                if checkpoint_path is not None:
                    self.fnet.save_model(checkpoint_path)

        # Train for remaining in the batch
        if len(batch > 0):
            self.fnet.train(batch, logging=logging, log_file=log_file)
            batch = [] # Empty the batch

            # Save the network
            if checkpoint_path is not None:
                self.fnet.save_model(checkpoint_path)


if __name__ == '__main__':
    # Create a player
    player = Player(13, 100, 10)
    player.self_play(10, 'networks/testing1.model', logging=True, log_file='logs/log_testing1.txt')

