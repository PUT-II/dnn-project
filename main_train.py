from os.path import isfile

import numpy as np

from udrl.setup_helper import SetupHelper
from udrl.train_params import TrainParams
from udrl.trainer import UdrlTrainer


def load_previous_train_data(trainer: UdrlTrainer):
    from udrl.replay_buffer import ReplayBuffer

    if not isfile('buffer_latest.npy') or not isfile('behavior_latest.npy') or not isfile('history_latest.npy'):
        return None

    buffer = ReplayBuffer()
    buffer.load('buffer_latest.npy')

    behavior = trainer.initialize_behavior_function()
    behavior.load('behavior_latest.pth', trainer.device)

    learning_history = list(np.load('history_latest.npy', allow_pickle=True))

    return buffer, behavior, learning_history


def train(resume_training: bool = False):
    envs = []
    for i in range(1, 8):
        envs += [SetupHelper.get_environment(world=i, stage=j) for j in range(1, 4)]
    device = SetupHelper.get_device()

    params = TrainParams(save_on_eval=True)
    trainer = UdrlTrainer(envs, device, params)

    data = load_previous_train_data(trainer) if resume_training else None
    if data is None:
        behavior, buffer, learning_history = trainer.train()
    else:
        behavior, buffer, learning_history = data
        behavior, buffer, learning_history = trainer.train(behavior, buffer, learning_history)

    behavior.save('behavior_latest.pth')
    buffer.save('buffer_latest.npy')
    np.save('history_latest.npy', learning_history)


if __name__ == "__main__":
    train(True)
