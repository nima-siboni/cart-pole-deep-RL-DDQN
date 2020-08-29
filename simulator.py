import numpy as np
import random
import os
import gym as gym
from PIL import Image
from tensorflow.keras.models import load_model
from agent import agent_learner
from utilfunctions import initializer
from utilfunctions import single_shape_adaptor
from utilfunctions import update_state_step
from utilfunctions import one_hot
from rl_utils import Histories


# create the environment
env = gym.make('CartPole-v0')

# creating the agent with the apropriate action and feature sizes
nr_features = env.observation_space.high.shape[0]
nr_actions = env.action_space.n

agent = agent_learner(nr_features=nr_features, nr_actions=nr_actions, gamma=0.98, stddev=0.05, learning_rate=0.0001)

# setting the random seeds
training_log = np.array([])
histories = Histories()

agent.Q_t = load_model('./training-results/Q-target/trained-agents/time-stamp-200')

figdir = './performance-and-animations/animations/trained/'

if not os.path.exists(figdir):
    os.makedirs(figdir)

initial_state = env.reset()

state, terminated, steps = initializer(initial_state)
state = single_shape_adaptor(state, nr_features)

while not terminated:

    action_id = agent.action_based_on_Q_target(state, env, epsilon=0)

    one_hot_action = one_hot(action_id, nr_actions)

    new_state, reward, terminated, info = env.step(action_id)

    new_state = single_shape_adaptor(new_state, nr_features)

    state, steps = update_state_step(new_state, steps)

    img = env.render(mode='rgb_array')
    
    img = Image.fromarray(img)
    if (steps < 10):
        timestamps = '00'+str(steps)
    else:
        if steps < 100:
            timestamps = '0'+str(steps)
        else:
            timestamps = str(steps)

    img.save(figdir+'state_'+timestamps+'.png')
