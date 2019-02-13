import os

import numpy as np
import tensorflow as tf


tf.enable_eager_execution()


# Functional parameters
SAVE_NAME = 'moonlander_ddpg'


class DDPG:
    def __init__(self):
        """Construct DDPG class based on global variables."""
        self.save_actor = os.path.join(os.getcwd(), 'model_states', 'ddpg', SAVE_NAME + '_actor.h5')
        self.save_critic = os.path.join(os.getcwd(), 'model_states', 'ddpg', SAVE_NAME + '_critic.h5')


if __name__ == '__main__':
    ddpg = DDPG()