from udrl.agent import UdrlAgent
from udrl.setup_helper import SetupHelper

env = SetupHelper.get_environment()
device = SetupHelper.get_device()

agent = UdrlAgent(env, device, 3)
agent.load_behavior('behavior_latest.pth')

done = False
agent.reset_env()
for step in range(5000):
    if done:
        break

    # state --> image
    # info --> information about mario
    action = agent.get_action(desired_return=300, horizon=400)
    _, done = agent.step(action)

    env.render()

env.close()
