import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class SetupHelper:

    @staticmethod
    def get_device() -> torch.device:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_environment() -> JoypadSpace:
        env = gym_super_mario_bros.make('SuperMarioBros-v1')
        return JoypadSpace(env, SIMPLE_MOVEMENT)
