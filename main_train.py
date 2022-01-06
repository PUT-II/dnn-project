import numpy as np

from udrl.setup_helper import SetupHelper
from udrl.train_params import TrainParams
from udrl.trainer import UdrlTrainer


def load_previous_train_data(trainer: UdrlTrainer):
    from udrl.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer()
    buffer.load('buffer_latest.npy')

    behavior = trainer.initialize_behavior_function()
    behavior.load('behavior_latest.pth')

    learning_history = list(np.load('history_latest.npy', allow_pickle=True))

    return buffer, behavior, learning_history


def train(resume_training: bool = False):
    env = SetupHelper.get_environment()
    device = SetupHelper.get_device()

    params = TrainParams(save_on_eval=True)
    trainer = UdrlTrainer(env, device, params)

    if resume_training:
        behavior, buffer, learning_history = load_previous_train_data(trainer)
        behavior, buffer, learning_history = trainer.train(behavior, buffer, learning_history)
    else:
        behavior, buffer, learning_history = trainer.train()

    behavior.save('behavior_latest.pth')
    buffer.save('buffer_latest.npy')
    np.save('history_latest.npy', learning_history)


if __name__ == "__main__":
    train(False)
