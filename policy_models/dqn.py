import gym

import numpy as np
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple
from itertools import count


# General parameters
RENDER = True
MODEL_DIR = os.path.join(os.getcwd(), 'models')
ENVIRONMENT = 'Pong-v0'
LOAD_MODEL = False


# Hyper parameters
EPS_START = 1.0
EPS_END = 0.5 # Cannot go lower than this value
ANNEAL_RANGE = 1000000
GAMMA = 0.999
STORE_N = 20000
MINIBATCH = 32
IN_H = 84
IN_W = 84
IN_SIZE = (IN_H, IN_W)
TARGET_UPDATE = 10
NUM_EPISODES = 2000
PROGRESS_STORAGE = 50


# Definition of replay memory
Transition = namedtuple('Transition', 
                        ('s0', 'a', 'r', 's1'))


class ReplayMemory:
    def __init__(self, capacity):
        """ Storage of transitions as named tuples. """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        transition = Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, n_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Environment defition
env = gym.make(ENVIRONMENT)
model_path = os.path.join(MODEL_DIR, (ENVIRONMENT+'.tar'))
n_actions = env.action_space.n


# Memory defintion
memory = ReplayMemory(STORE_N)


# Network definition
dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy_net = Net(n_actions).to(device)
target_net = Net(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Make sure the networks are initialized the same, target_net will retain previous state of policy policy_net.
criterion = nn.MSELoss()
optimizer = optim.SGD(policy_net.parameters(), lr=0.001, momentum=0.9)


# Preprocessing definition
im_transform = T.Compose([T.ToPILImage(),
                         T.Grayscale(),
                         T.Resize(IN_SIZE),
                         T.ToTensor()])


def preprocess(x):
    """ Applies image transformation and adds dimension. """
    x = im_transform(x)
    return x.unsqueeze(0)


# Definition of action sampling function
steps_done = 0


def sample_action(state):
    """ Sample action according to epsilon greedy strategy. Epsilon decays over first 100000 frames. """
    global steps_done
    sample = random.uniform(0, 1)
    if steps_done < ANNEAL_RANGE:
        epsilon = EPS_START - (EPS_START - EPS_END) * steps_done / 1000000
    else:
        epsilon = EPS_END
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(load_model=False):
    """ Function that performs a single single step of the optimization. """
    # Return in case not enough samples in memory yet
    if len(memory) < MINIBATCH:
        return

    # Sample transition and extract desired data into separate variables
    transition_samples = memory.sample(MINIBATCH)
    batch = Transition(*zip(*transition_samples))
    
    # Determine not terminal next states
    non_terminal_mask = torch.tensor([s is not None for s in batch.s1], dtype=torch.uint8, device=device)
    non_final_next_states = torch.cat([s for s in batch.s1 if s is not None])

    # Build tensors out of the batch data
    batch_s0 = torch.cat(batch.s0)
    batch_r = torch.cat(batch.r)
    batch_a = torch.cat(batch.a)

    # Determine policy_net output
    state_action_values = policy_net.forward(batch_s0).gather(1, batch_a)

    # Determine target_net output
    next_state_values = torch.zeros(MINIBATCH, device=device)
    next_state_values[non_terminal_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_values = (next_state_values * GAMMA) + batch_r

    # Determine loss and set target net equal to policy net
    loss = criterion(state_action_values, expected_state_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return state_action_values


def track_training_progress(rolling_Q, state_action_values):
    """ Print progress of learning to terminal """
    if rolling_Q.item() == 0:
        rolling_Q = state_action_values.mean()
    else:
        rolling_Q = rolling_Q * 0.99 + state_action_values.mean() * 0.01
    return rolling_Q
    

def track_training_progress(states):
    """ Print progress of learning to terminal """
    actions = policy_net(states).max(1)[0].detach()
    av_Q = actions.sum() / PROGRESS_STORAGE
    print("Average maximum Q value for fixed states is: {}".format(av_Q))


if __name__ == '__main__':
    # Load model if specified in hyper parameters
    if LOAD_MODEL:
        checkpoint = torch.load(model_path)
        policy_net.load_state_dict(checkpoint['policy_state_dict'])
        target_net.load_state_dict(checkpoint['policy_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        memory = checkpoint['memory']
        ep_start = checkpoint['episode']
        policy_net.train()
        target_net.train()
        print('Model loaded & ready for training')
    else:
        ep_start = 0

    episodes_duration = []
    prev_screen = None
    rolling_Q = torch.zeros(1) 
    progress_tracking_mem = torch.tensor([], device=device)
    for i_episode in range(ep_start, NUM_EPISODES):
        # Define current state as difference between current and previous frame
        current_screen = preprocess(env.reset()).to(device)
        state = current_screen - prev_screen if prev_screen is not None else torch.zeros(1, 1, IN_H, IN_W, device=device)
        for t in count():
            if RENDER:
                env.render()
                time.sleep(0.002)
            action = sample_action(state)

            # Apply action and observe state advancement
            prev_screen = current_screen
            current_screen, reward, done, info = env.step(action.item())
            current_screen = preprocess(current_screen).to(device)
            reward = torch.tensor([reward], device=device)

            # Build next state
            if not done:
                next_state = current_screen - prev_screen
            else:
                next_state = None

            # Store transition in memory 
            memory.push(state, action, reward, next_state)
            if progress_tracking_mem.size()[0] < PROGRESS_STORAGE:
                progress_tracking_mem = torch.cat((progress_tracking_mem, state))

            # Move to next state
            state = next_state

            # Perform optimization on the policy network
            state_action_values = optimize_model()
            if done:
                episodes_duration.append(t + 1)
                break

        print('Memory size {}'.format(len(memory)))
        print('Episode {} done! \n'.format(i_episode+1))
        if progress_tracking_mem.size()[0] == PROGRESS_STORAGE:
            track_training_progress(progress_tracking_mem)

        # Update target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save model
        if i_episode % 10 == 0:
            torch.save({
                'policy_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': i_episode,
                'memory': memory 
            }, model_path)