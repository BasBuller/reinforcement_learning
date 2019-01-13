import tensorflow as tf
import gym
import pickle
import os


# Define baseline policy gradient class
class Policy:
    def __init__(self, model, environment, model_save, class_save, *args, **kwargs):
        """ Base class for policy gradient algorithms.
        
        In: 
            - model: tensorflow policy function approximator
            - environment: OpenAI gym environment used
            - model_save: name of the save file for tf model
            - class_save: name of the save file for the class
        """
        self.model = model
        self.env = gym.make(environment)
        self.model_path = os.path.join(os.getcwd(), 'models', model_save)
        self.class_path = os.path.join(os.getcwd(), 'models', class_save)

    def save(self):
        """ Save the complete policy instance. """
        # Cannot save environment
        temp_env = self.env
        self.env = None

        # Save model and class
        self.model.save(self.model_path)
        with open(self.class_path, 'wb') as p_file:
            pickle.dump(self, p_file)

        # Set back enviroment
        self.env = temp_env

    def load(self):
        self.model = tf.keras.models.load_model(self.save_path)
