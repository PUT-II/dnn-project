import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from udrl.behavior import Behavior
from udrl.udrl import UDRL

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
behavior = Behavior(
    action_size=env.action_space.n,
    info_size=3,
    device=device,
    command_scale=[0.02, 0.01]
)

behavior.load('behavior_latest.pth')

agent = UDRL(env, device)

done = False
state = agent.preprocess_state(env.reset())
info = [0.0, 0.0, 0.0]
for step in range(5000):
    if done:
        break

    # state --> image
    # info --> information about mario
    action = agent.get_action(behavior.action, state, [300, 400], info)
    state, _, done, info = agent.step(action)

    env.render()

env.close()
