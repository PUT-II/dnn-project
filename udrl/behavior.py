import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from torch.distributions import Categorical
from torch.optim import Adam


class Behavior(nn.Module):
    def __init__(
            self,
            state_size,
            action_size,
            device,
            command_scale: list = None
    ):
        if command_scale is None:
            command_scale = [1, 1]

        super().__init__()

        # TODO: Investigate neural network with multiple inputs (state and info)

        self.command_scale = torch.FloatTensor(command_scale).to(device)

        self.state_fc = nn.Sequential(nn.Linear(state_size, 64),
                                      nn.Tanh())

        self.command_fc = nn.Sequential(nn.Linear(2, 64),
                                        nn.Sigmoid())

        self.output_fc = nn.Sequential(nn.Linear(64, 128),
                                       nn.ReLU(),
                                       # nn.Dropout(0.2),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       # nn.Dropout(0.2),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, action_size))

        self.to(device)

    def forward(self, state, command):
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.command_scale)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)

    def action(self, state, command):
        logits = self.forward(state, command)
        probs = nn_functional.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().item()

    def greedy_action(self, state, command):
        logits = self.forward(state, command)
        probs = nn_functional.softmax(logits, dim=-1)
        return np.argmax(probs.detach().cpu().numpy())

    def init_optimizer(self, optim=Adam, lr=0.003):
        # noinspection PyAttributeOutsideInit
        self.optim = optim(self.parameters(), lr=lr)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
