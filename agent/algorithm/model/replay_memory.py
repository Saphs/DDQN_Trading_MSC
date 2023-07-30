from collections import namedtuple
import random

from torch import Tensor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state_batch: Tensor, action_batch: Tensor, next_state_batch: Tensor, reward_batch: Tensor):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        #  ToDo: Properly define how replay memory will work with the new mini-batches.
        #   How many transitions are kept etc. Currently only the first transition is kept.
        #print(f"STATE_BATCH: {state_batch}")
        #print(f"ACTION_BATCH: {action_batch}")
        #print(f"NEXT_STATE_BATCH: {next_state_batch}")
        #print(f"REWARD_STATE_BATCH: {reward_batch}")
        self.memory[self.position] = Transition(state_batch, action_batch, next_state_batch, reward_batch)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, 1)

    def __len__(self):
        return len(self.memory)