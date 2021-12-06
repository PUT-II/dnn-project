import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from udrl.behavior import Behavior
from udrl.udrl import UDRL

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
behavior = Behavior(state_size=10,
                    action_size=env.action_space.n,
                    device=device,
                    command_scale=[0.02, 0.01])

behavior.load('behavior_mario.pth')

done = False
env.reset()
_, reward, _, state = env.step(0)
state = UDRL.preprocess_state(state)
for step in range(5000):
    if done:
        env.reset()
        _, reward, done, state = env.step(0)
        state = UDRL.preprocess_state(state)

    # state = image
    # info = information about player
    state_input = torch.FloatTensor(state).to(device)
    command_input = torch.FloatTensor([302, 400]).to(device)
    action = behavior.action(state_input, command_input)
    _, _, done, state = env.step(action)
    state = UDRL.preprocess_state(state)

    env.render()

env.close()
