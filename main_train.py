import numpy as np

from udrl.setup_helper import SetupHelper
from udrl.train_params import TrainParams
from udrl.trainer import UdrlTrainer

env = SetupHelper.get_environment()
device = SetupHelper.get_device()

params = TrainParams(save_on_eval=True)
trainer = UdrlTrainer(env, device, params)

behavior, buffer, learning_history = trainer.train()

behavior.save('behavior_mario.pth')
buffer.save('buffer_mario.npy')
np.save('history_mario.npy', learning_history)
