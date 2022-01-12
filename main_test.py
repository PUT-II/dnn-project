import numpy as np

from udrl.agent import UdrlAgent
from udrl.setup_helper import SetupHelper
from udrl.util import clip_reward

env = SetupHelper.get_environment(world=1)
device = SetupHelper.get_device()

agent = UdrlAgent(env, device, info_size=1)
agent.load_behavior('behavior_latest.pth', device)
agent.reset_env()

done = False
desired_return = 2500
desired_horizon = 330
all_rewards = []
for step in range(20000):
    if done:
        break

    action = agent.get_action(desired_return, desired_horizon)
    reward, done = agent.step(action)
    all_rewards.append(reward)
    env.render()

    desired_horizon = np.random.randint(200, 400)
    desired_return = clip_reward(desired_return - reward, desired_horizon * -15, desired_horizon * 15)

total_reward = sum(all_rewards)
print(f"Total reward: {total_reward}")
env.close()
