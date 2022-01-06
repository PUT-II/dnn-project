from udrl.agent import UdrlAgent
from udrl.setup_helper import SetupHelper
from udrl.util import clip_reward

env = SetupHelper.get_environment()
device = SetupHelper.get_device()

agent = UdrlAgent(env, device, 3)
agent.load_behavior('behavior_latest.pth')

done = False
agent.reset_env()
desired_return = 75
desired_horizon = 150
for step in range(5000):
    if done:
        break

    # state --> image
    # info --> information about mario
    action = agent.get_action(desired_return, desired_horizon)
    reward, done = agent.step(action)
    env.render()

    delta_return = desired_return - reward
    desired_return = clip_reward(desired_return - reward, -15, 15)

env.close()
