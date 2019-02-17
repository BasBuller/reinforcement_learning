import time
import os
import pickle

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Functionality flags
RENDER = False
LOAD_MODEL = True
SAVE_NAME = 'cartpole_ppo'


# Environment defintions
ENVIRONMENT = 'CartPole-v0'
N_ACTIONS = 2
OBSERVATIONS = 4
DUMMY_ADVANTAGE = np.zeros((1, 1))
DUMMY_POLICY = np.zeros((1, N_ACTIONS))


# Hyper parameters
ARGMAX_ACTION = False
ACTIVATION_FUNC = 'tanh'
N_ACTORS = 1
M_BATCH_SIZE = 32 * N_ACTORS
T_STEPS = 128
K_EPOCHS = 3
GAMMA = 0.99
LAMBDA = 0.95
EPISODES = 1000
ADAM_STEP = 2.5e-4
EPSILON = 0.1
ALPHA_START = 1
ALPHA_END = 0 


def actor_loss_func(previous_policy, advantage):
    def loss_func(y_true, y_pred):
        """
        y_true: a_t, action taken at time t.
        y_pred: current policy.
        """
        prev_prob = previous_policy * y_true
        curr_prob = y_pred * y_true
        r = curr_prob / (prev_prob + 1e-10)
        loss = -tf.reduce_mean(tf.minimum(r, tf.clip_by_value(r, 1 - EPSILON, 1 + EPSILON)) * advantage)
        return loss
    return loss_func


def construct_actor():
    """Construct and return an actor neural network."""
    inputs = tf.keras.Input(shape=(OBSERVATIONS,))
    old_prediction = tf.keras.Input((N_ACTIONS,))
    advantage = tf.keras.Input((1,))

    # Network structure
    x = layers.Dense(64, activation=ACTIVATION_FUNC)(inputs)
    x = layers.Dense(64, activation=ACTIVATION_FUNC)(x)
    outputs = layers.Dense(N_ACTIONS, activation='softmax')(x)

    # Construct and compile network
    actor_loss = actor_loss_func(old_prediction, advantage)
    model = tf.keras.Model(inputs=[inputs, old_prediction, advantage], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=ADAM_STEP),
                  loss=actor_loss)
    model.summary()
    return model


def construct_critic():
    """Construct and return a critic neural network."""
    # Network structure
    inputs = tf.keras.Input(shape=(OBSERVATIONS,))
    x = layers.Dense(64, activation=ACTIVATION_FUNC)(inputs)
    x = layers.Dense(64, activation=ACTIVATION_FUNC)(x)
    outputs = layers.Dense(1)(x)

    # Construct and compile network
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=ADAM_STEP),
                  loss='mse')
    model.summary()
    return model


class PPO:
    def __init__(self):
        """Construct PPO class based on global variables."""
        self.save_actor = os.path.join(os.getcwd(), 'model_states', 'ppo', SAVE_NAME + '_actor_weights.h5')
        self.save_critic = os.path.join(os.getcwd(), 'model_states', 'ppo', SAVE_NAME + '_critic_weights.h5')
        self.actor = construct_actor()
        self.critic = construct_critic()
        if LOAD_MODEL:
            self.actor.load_weights(self.save_actor)
            self.critic.load_weights(self.save_critic)

    def save(self):
        """Save model's current state."""
        self.actor.save_weights(self.save_actor)
        self.critic.save_weights(self.save_critic)

    def sample_action(self, action_probs):
        """Sample an action based of the provided probabilities."""
        if ARGMAX_ACTION:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(N_ACTIONS, p=action_probs.flatten())
        action_vector = np.zeros(N_ACTIONS)
        action_vector[action] = 1
        return action, action_vector

    def general_advantage_estimation(self, rewards, state_values):
        """Calculate GAE value for T timesteps."""
        true_state_values = np.zeros((len(state_values),))
        # true_state_values[-1] = state_values[-1].flatten()
        for i in reversed(range(len(rewards))):
            true_state_values[i] = rewards[i] + GAMMA * true_state_values[i + 1]
        advantages = true_state_values[:-1] - np.concatenate(state_values[:-1]).flatten()
        return true_state_values[:-1].reshape((-1, 1)), advantages.reshape((-1, 1))

    def telescoping_reward(self, rewards):
        for idx in reversed(range(len(rewards) - 1)):
            rewards[idx] += rewards[idx + 1] * GAMMA
        return rewards

    def run_episode(self):
        """Run one epsiode and collect all needed information."""
        env = gym.make(ENVIRONMENT)
        prev_observation = env.reset()
        prev_observation = prev_observation.reshape((1, -1))
        observations_episode    = []
        state_values_episode    = [] 
        action_probs_episode    = []
        actions_episode         = []
        true_state_values_episode         = []
        temp_rewards            = []

        # Run for T timesteps and collect observations_episode, true_state_values_episode
        for _ in range(T_STEPS):
            if RENDER:
                env.render()
                time.sleep(0.005)

            # Sample state value and action based on previous observation
            state_value = self.critic.predict(prev_observation)
            action_probs = self.actor.predict([prev_observation, DUMMY_POLICY, DUMMY_ADVANTAGE])
            action, action_vector = self.sample_action(action_probs)

            # Run environment with action
            observation, reward, done, _ = env.step(action)

            # Append collected data
            observations_episode.append(prev_observation)
            temp_rewards.append(reward)
            state_values_episode.append(state_value)
            action_probs_episode.append(action_probs)
            actions_episode.append(action_vector)

            # If environment instance is over reset
            if done:
                true_state_values_episode += self.telescoping_reward(temp_rewards)
                temp_rewards = []
                prev_observation = env.reset()
            else:
                prev_observation = observation
            prev_observation = prev_observation.reshape((1, -1))

        # Last processing and return
        true_state_values_episode += self.telescoping_reward(temp_rewards)
        state_values_episode.append(self.critic.predict(prev_observation))
        return observations_episode, true_state_values_episode, state_values_episode, action_probs_episode, actions_episode

    def train(self):
        for episode in range(EPISODES):
            # Collect results for single episode
            observations_episode, true_state_values_episode, state_values_episode, action_probs_episode, actions_episode = self.run_episode()
            observations_episode = np.vstack(observations_episode)
            action_probs_episode = np.vstack(action_probs_episode)
            actions_episode = np.vstack(actions_episode)
            advantages_episode = np.array(true_state_values_episode).reshape((-1, 1)) - np.vstack(state_values_episode[:-1])

            # Update networks
            callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), 'logs', 'ppo'))]
            self.critic.fit(observations_episode, 
                            true_state_values_episode, 
                            epochs=K_EPOCHS, 
                            batch_size=M_BATCH_SIZE,
                            callbacks=callbacks)
            self.actor.fit([observations_episode, action_probs_episode, advantages_episode], 
                           actions_episode,
                           epochs=K_EPOCHS,
                           batch_size=M_BATCH_SIZE,
                           callbacks=callbacks)

            print('Episode {} done!'.format(episode))
            if (episode + 1) % 10 == 0:
                self.save()


if __name__ == '__main__':
    ppo = PPO()
    ppo.train()