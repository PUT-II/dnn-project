import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from udrl import udrl
from udrl.train_params import TrainParams

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

params = TrainParams()
algorithm = udrl.UDRL(env, device, params)

behavior, buffer, learning_history = algorithm.train()

behavior.save('behavior_mario.pth')
buffer.save('buffer_mario.npy')
np.save('history_mario.npy', learning_history)
