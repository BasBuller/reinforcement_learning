""" Defines and trains a Pong Reinforcement Learning bot using the OpenAI gym environtment. """
import gym
import numpy as np
import pickle


# Hyper Parameters
H = 200  # Number of hidden units
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
resume = True # Resume from previous checkpoint
render = False


# Model Initialization
input_size = 80 * 80  # Input is 80 by 80 grid
if resume:
    with open('save.p', 'rb') as file_handle:
        model = pickle.load(file_handle)
        print("Loaded previous model")
else:
    model = {}
    model['W1'] = np.random.rand(H, input_size) / np.sqrt(input_size)
    model['W2'] = np.random.rand(H) / np.sqrt(H)
    print("created new model")

grad_buffer = {k : np.zeros_like(v) for k,v in model.items()}
rmsprop_cache = {k : np.zeros_like(v) for k,v in model.items()}

def preprocess(Img):
    """ Crop [210x160x3] image to 80x80 for faster computation """
    Img = Img[35:195]
    Img = Img[::2, ::2, 0]
    Img[Img == 114] = 0
    Img[Img == 109] = 0
    Img[Img != 0] = 1
    return Img.astype(np.float).ravel()


def discounted_rewards(r):
    """ Discount rewards based on gamma value """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(r))):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def sigmoid(x):
    """ Simple sigmoid function to squash log probability to [0,1] range. """
    return 1 / (1 + np.exp(-x))


def policy_forward(x):
    """ Evaluation of the agent's policy. Makes use of a ReLU activation for the hidden layer and sigmoid function
    to squash output to probability range [0,1]. """
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    log_p = np.dot(model['W2'], h)
    p = sigmoid(log_p)
    return p, h


def policy_backward(eph, epdlogp):
    """ Updating of the model weights based on RMSprop. """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[dh <= 0] = 0
    dW1 = np.dot(eph.T, epx)
    return {'W1': dW1, 'W2': dW2}


# Define pong environment
env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = 0
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()

    # Preprocess inputs
    cur_x = preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(input_size)
    prev_x = cur_x

    # forward pass
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 

    # keeps logs and determine gradient
    xs.append(x)
    hs.append(h)
    y = 1 if action is 2 else 0
    dlogps.append(y - aprob)

    # Perform action and gather reward
    cur_x, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)

    if done:
        episode_number += 1

        # Reformat state, reward, input, output
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []

        # Discount reward and normalize
        discounted_epr = discounted_rewards(epr) 
        discounted_epr -= np.linalg.norm(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # Scale gradient with reward, RL magic here
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]

        # Perform RMSprop to update weights
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / np.sqrt(rmsprop_cache[k]+ 1e-5) 
                grad_buffer[k] = np.zeros_like(v)

        # Bookkeeping
        running_reward = reward_sum if running_reward == 0 else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was {}. running mean: {}'.format(reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()
        prev_x = 0
    
    if reward != 0:
        print('ep {}: game finished, reward: {}'.format(episode_number, reward) + ('' if reward == -1 else '!!!!!!!!!'))
