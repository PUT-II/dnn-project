import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from udrl.behavior import Behavior
from udrl.udrl import UDRL

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
behavior = Behavior(action_size=env.action_space.n,
                    device=device,
                    command_scale=[0.02, 0.01])

behavior.load('behavior_mario.pth')

agent = UDRL(env, device)

done = False
env.reset()
state, reward, _ = agent.step(0)  # NOOP
for step in range(5000):
    if done:
        env.reset()
        state, reward, done = agent.step(0)  # NOOP

    # state = image
    # info = information about player
    action = agent.get_action(behavior.action, state, [302, 400])
    state, _, done = agent.step(action)

    env.render()

env.close()
