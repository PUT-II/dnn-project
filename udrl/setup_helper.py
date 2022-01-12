import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace


class SetupHelper:

    @staticmethod
    def get_device() -> torch.device:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_environment(world: int = 1, stage: int = 1) -> JoypadSpace:
        env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v1')
        return JoypadSpace(env, RIGHT_ONLY)
