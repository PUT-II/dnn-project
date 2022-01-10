import numpy as np

from udrl.agent import UdrlAgent
from udrl.setup_helper import SetupHelper
from udrl.util import clip_reward

env = SetupHelper.get_environment()
device = SetupHelper.get_device()

agent = UdrlAgent(env, device, 3)
agent.load_behavior('behavior_latest.pth', device)
agent.reset_env()

done = False
desired_return = 250
desired_horizon = 330
for step in range(5000):
    if done:
        break

    action = agent.get_action(desired_return, desired_horizon)
    reward, done = agent.step(action)
    env.render()

    desired_horizon = np.random.randint(200, 400)
    desired_return = clip_reward(desired_return - reward, desired_horizon * -15, desired_horizon * 15)

env.close()
