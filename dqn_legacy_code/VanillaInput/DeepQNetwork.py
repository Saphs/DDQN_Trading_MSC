import torch.nn as nn
from torch.nn import Sequential


class NeuralNetwork(nn.Module):
    def __init__(self, state_length, action_length):
        super(NeuralNetwork, self).__init__()
        self.policy_network: Sequential = nn.Sequential(
            nn.Linear(state_length, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_length))

    def forward(self, x):
        return self.policy_network(x)
