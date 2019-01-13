import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np
import os
import pickle


LOAD_MODEL = False
RENDER = False
N_STEPS = 100


class PPO:
    def __init__(self, environment, network, model_name):
        self.lr = 0.01
        self.network = network 
        self.env = construct_environment(environment)
        self.model_dir = os.path.join(os.getcwd, 'models', model_name)
        self.model_name = model_name
        self.model_path = os.path.join(self.model_dir, self.model_name+'_tf.h5')
        self.dict_path = os.path.join(self.model_dir, self.model_name+'_param.p')
        
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def save(self, state_dict):
        self.network.save(self.model_path)
        with open(self.dict_path, 'wb') as file_:
            pickle.dump(state_dict, file_)

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path)
        state_dict = pickle.load(self.dict_path)
        # TODO: Process state dict by assigning its items to class parameters

    def sample_action(self, state):
        """ Sample an action using the policy network, based on current state. """
        return

    def train(self, n_episodes):
        # set seed and load model
        set_seeds(543)
        if LOAD_MODEL:
            self.load()
        
        episode_start = 0
        for episode in range(episode_start, n_episodes):
            return 


def set_seeds(seed):
    """ Sets global seed for numpy and tensorflow """
    tf.set_random_seed(seed)
    np.random.seed(seed)    


def construct_model(n_in, n_out, optimizer, loss_func, metrics):
    """ Simple model with 2 hidden layers of 64 units, has variable input.

    Arguments
    :n_in:      Number of input nodes
    :n_out:     Number of output nodes
    :optimizer: Tensorflow optimizer used in model
    :loss_func: Tensorflow loss function used in model
    :metrics:   Tensorflow metrics used to track training performance
    """
    inputs = keras.Input(shape=(n_in,))
    x = keras.layers.Dense(64, activation='tanh')(inputs)
    x = keras.layers.Dense(64, activation='tanh')(x)
    predictions = keras.layers.Dense(n_out, activation='softmax')(x)

    model = keras.Model(input=inputs, output=predictions)
    model.compile(optimizer=optimizer,
                  loss_func=loss_func,
                  metrics=metrics)
    return model


def construct_environment(env):
    return gym.make(env)