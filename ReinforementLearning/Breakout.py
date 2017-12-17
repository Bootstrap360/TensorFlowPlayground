# TensorFlow Tutorial #16 Reinforcement Learning

# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb
# https://www.youtube.com/watch?v=Vz5l886eptw

# Use Q Values

# Q is a table that maps a state to an action
# Q values start at zero and are updated to give positive or negative feedback

# r_t = reward
# gamma = discount-factor. This causes more distant reqards to contribute less tot he Q-value, thus making the agent favour rewards that are closer in time.

# Q [s_t, a_t] = r_t + gamma * max( Q[s_t+1, a])

# Need a function that will estimate the Q-values for the state. Use a Convolutional Neural Netowrk for this. (CNN)
# The Neural Network used in this implementation has 3 convolutional layers, all of which have filter-size 3x3. The layers have 16, 32, and 64 output channels, respectively. The stride is 2 in the first two convolutional layers and 1 in the last layer.
# Following the 3 convolutional layers there are 4 fully-connected layers each with 1024 units and ReLU-activation. Then there is a single fully-connected layer with linear activation used as the output of the Neural Network.
# This architecture is different from those typically used in research papers from DeepMind and others. They often have large convolutional filter-sizes of 8x8 and 4x4 with high stride-values. This causes more aggressive down-sampling of the game-state images. They also typically have only a single fully-connected layer with 256 or 512 ReLU units.
# During the research for this tutorial, it was found that smaller filter-sizes and strides in the convolutional layers, combined with several fully-connected layers having more units, were necessary in order to have sufficiently accurate Q-values. The Neural Network architectures originally used by DeepMind appear to distort the Q-values quite significantly. A reason that their approach still worked, is possibly due to their use of a very large Replay Memory with 1 million states, and that the Neural Network did one mini-batch of training for each step of the game-environment, and some other tricks.
# The architecture used here is probably excessive but it takes several days of training to test each architecture, so it is left as an exercise for the reader to try and find a smaller Neural Network architecture that still performs well.

import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import math


import reinforcement_learning as rl

env_name = 'Breakout-v0'


rl.checkpoint_base_dir = 'checkpoints_tutorial16/'
rl.update_paths(env_name=env_name)

agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=False)
model = agent.model
replay_memory = agent.replay_memory
agent.run(num_episodes=1)


