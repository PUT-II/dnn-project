import numpy as np
import torch

from udrl.behavior import Behavior
from udrl.util import get_state_size, preprocess_state, preprocess_info, get_state_channels


class UdrlAgent:

    def __init__(self, environment, device, info_size: int, command_scale: list = None):
        self.environment = environment
        self.info_size = info_size
        self.current_state: np.ndarray = np.zeros(get_state_size(), dtype=np.int32)
        self.current_info: np.ndarray = np.zeros((info_size,), np.float32)

        if command_scale is None:
            command_scale = [0.02, 0.01]

        self.device = device

        self.behavior = Behavior(
            action_size=environment.action_space.n,
            info_size=info_size,
            device=device,
            state_channels=get_state_channels(),
            command_scale=command_scale,
        )

    def get_action(self, desired_return: int, horizon: int) -> int:
        command = [desired_return, horizon]
        if get_state_channels() == 1:
            expanded_state = np.ascontiguousarray(np.expand_dims(self.current_state, axis=(0, 1)))
        else:
            expanded_state = np.ascontiguousarray(np.expand_dims(self.current_state, axis=(0,)))

        state_input = torch.FloatTensor(expanded_state).to(self.device)
        info_input = torch.FloatTensor(self.current_info).to(self.device)
        command_input = torch.FloatTensor(command).to(self.device)

        action = self.behavior.action(state_input, command_input, info_input)
        return action

    def step(self, action: int):
        state, reward, done, info = self.environment.step(action)
        self.current_state = preprocess_state(state)
        self.current_info = preprocess_info(info)
        return reward, done

    def reset_env(self):
        self.current_state = preprocess_state(self.environment.reset())
        self.current_info: np.ndarray = np.zeros((self.info_size,), np.float32)

    def load_behavior(self, file_path: str, device: torch.device):
        self.behavior.load(file_path, device)
