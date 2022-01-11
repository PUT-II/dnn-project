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
            state_channels: int = 1,
            command_scale: list = None
    ):
        if command_scale is None:
            command_scale = [1, 1]

        super().__init__()

        self.command_scale = torch.FloatTensor(command_scale).to(device)

        # noinspection PyTypeChecker
        self.state_fc = nn.Sequential(
            nn.Conv2d(in_channels=state_channels, out_channels=16, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.Flatten()
        )

        self.command_fc = nn.Sequential(nn.Linear(2, 384),
                                        nn.Sigmoid())

        self.info_fc = nn.Sequential(nn.Linear(info_size, 384),
                                     nn.Sigmoid())

        self.output_fc = nn.Sequential(nn.Linear(384, 128),
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

    def init_optimizer(self, optim=Adam, lr=0.003):
        # noinspection PyAttributeOutsideInit
        self.optim = optim(self.parameters(), lr=lr, eps=1e-4)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device: torch.device):
        self.load_state_dict(torch.load(filename, map_location=device))
