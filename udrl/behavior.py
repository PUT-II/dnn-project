import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from torch import FloatTensor
from torch.distributions import Categorical
from torch.optim import Adam


class Behavior(nn.Module):
    def __init__(
            self,
            action_size: int,
            info_size: int,
            device,
            command_scale: list = None
    ):
        if command_scale is None:
            command_scale = [1, 1]

        super().__init__()

        self.command_scale = torch.FloatTensor(command_scale).to(device)

        # noinspection PyTypeChecker
        self.state_fc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=7680, out_features=512)
        )

        self.command_fc = nn.Sequential(nn.Linear(2, 512),
                                        nn.Sigmoid())

        self.info_fc = nn.Sequential(nn.Linear(info_size, 512),
                                     nn.Sigmoid())

        self.output_fc = nn.Sequential(nn.Linear(512, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, action_size))

        self.to(device)

    def forward(self, state: FloatTensor, command: FloatTensor, info: FloatTensor):
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.command_scale)
        info_output = self.info_fc(info)
        embedding = torch.mul(state_output, command_output)
        embedding = torch.mul(embedding, info_output)
        return self.output_fc(embedding)

    def action(self, state, command, info):
        logits = self.forward(state, command, info)
        probs = nn_functional.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().item()

    def greedy_action(self, state, command, info):
        logits = self.forward(state, command, info)
        probs = nn_functional.softmax(logits, dim=-1)
        return np.argmax(probs.detach().cpu().numpy())

    def init_optimizer(self, optim=Adam, lr=0.003):
        # noinspection PyAttributeOutsideInit
        self.optim = optim(self.parameters(), lr=lr, eps=1e-4)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
