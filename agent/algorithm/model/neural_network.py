import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential


class NeuralNetwork(nn.Module):

    # The action space defined the number of outputs possible - out use case only requires 3: "Buy", "hold" and "Sell"
    ACTION_SPACE = 3
    print_active = False

    def __init__(self, state_length):
        super(NeuralNetwork, self).__init__()

        self.policy_network: Sequential = nn.Sequential(
            nn.Linear(state_length, 128),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.ACTION_SPACE)
        )

        """ Large net, Better?
        self.policy_network: Sequential = nn.Sequential(
            nn.Linear(state_length, 128),
            #nn.LayerNorm(128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.ACTION_SPACE)
        )
        
        self.l1 = nn.Linear(state_length, 128)
        self.l2 = nn.Linear(128, 256)
        self.b23 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(256, self.ACTION_SPACE)
        """


    def forward(self, x):
        """
        x = F.relu(self.l1(x))
        x = F.relu(self.b23(self.l2(x)))
        if self.print_active:
            print(x[1])
        x = F.relu(self.l3(x))
        """

        return self.policy_network(x)

