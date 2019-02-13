import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Net(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return action_scores, state_values


def sample_action(state):
    """ Sample an action using the policy network, based on current state. """
    state = torch.from_numpy(state).float().unsqueeze(0)
    log_probs, state_value = act_crit(state)
    sample = random.uniform(0, 1)
    if steps_done < ANNEAL_RANGE:
        epsilon = EPS_START - (EPS_START - EPS_END) * steps_done / ANNEAL_RANGE
    else:
        epsilon = EPS_END
    if sample > epsilon:
        with torch.no_grad():
            action = log_probs.max(1)[1].item()
    else:
        with torch.no_grad():
            action = log_probs.min(1)[1].item()
    act_crit.saved_actions.append(SavedAction(log_probs[0, action], state_value))
    return action


def run_episode():
    """ Execute an entire episode for a single instance. """
    state_0 = env.reset()
    for t in range(MAX_T):
        # Run till done, or max number of time steps reached
        action = sample_action(state_0)
        observation, reward, done, info = env.step(action)
        if RENDER:
            env.render()
        reward = reward if done else 0
        act_crit.rewards.insert(0, reward)
        if done:
            break

    # Process rewards and determine losses
    R = 0
    saved_actions = act_crit.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in act_crit.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps.item())
    for (log_prob, value), r in zip(saved_actions, rewards):
        td = r - value.item()
        policy_losses.append(-log_prob * td)  # Use -log_prob because built in methods are descendant instead of ascendant
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    act_crit.rewards = []
    act_crit.saved_actions = []
    return loss, t


def update_network(loss):
    """ Collects results from all instances of the actor critic network and updates weights. """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    # Hyper parameters
    SEED = 543
    GAMMA = 0.99
    MAX_T = 1000
    EPS_START = 0.5
    EPS_END = 0.05
    ANNEAL_RANGE = 1000000


    # Bookkeeping and system parameters
    RENDER = False
    LOAD_MODEL = False
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 25000 
    SAVE_PATH = 'CartPole.tar'
    PROCESSES = 2


    # Environment and network initiation
    env = gym.make('CartPole-v0')
    act_crit = Net(4, 2)
    act_crit.share_memory()
    torch.manual_seed(SEED)
    optimizer = optim.RMSprop(act_crit.parameters(), centered=False)


    # For annealing epsilon
    steps_done = 0
    

    if LOAD_MODEL:
        checkpoint = torch.load(SAVE_PATH)
        act_crit.load_state_dict(checkpoint['act_crit'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        i_episode = checkpoint(['i_episode'])
        running_reward = checkpoint(['running_reward'])
        act_crit.train()
    else:
        i_episode = 1
        running_reward = 10


    # Seed definitions
    env.seed(SEED)
    torch.manual_seed(SEED)


    for i_episode in range(i_episode, 500000001):
        loss, t = run_episode()
        update_network(loss)
        running_reward = 0.99 * running_reward + 0.01 * t


        if i_episode % LOG_INTERVAL == 0:
            print('Episode {}\tLates Duration: {}\tRunning Reward: {}'.format(i_episode, t, running_reward))


        if i_episode % SAVE_INTERVAL == 0:
            torch.save({
                'act_crit': act_crit.state_dict(),
                'optimizer': optimizer.state_dict(),
                'i_episode': i_episode,
                'running_reward': running_reward
            }, SAVE_PATH)