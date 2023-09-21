from collections import namedtuple
import random

import torch
from torch import Tensor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state_batch: Tensor, action_batch: Tensor, next_state_batch: Tensor, reward_batch: Tensor):
        """Saves a transition."""
        if next_state_batch is None:
            # We chicken out and dont deal with terminal states in the replay memory to reduce complexity
            return
        transition_tuples = list(zip(state_batch, action_batch, next_state_batch, reward_batch))
        for tran in transition_tuples:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*tran)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        tuples = [*zip(*transitions)]
        s = torch.stack(tuples[0])
        a = torch.stack(tuples[1])
        s_prime = torch.stack(tuples[2])
        r = torch.stack(tuples[3])
        return s, a, s_prime, r

    def __len__(self):
        return len(self.memory)