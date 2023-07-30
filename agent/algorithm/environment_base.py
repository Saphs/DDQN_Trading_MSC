from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import math

from numpy import ndarray
from torch import Tensor


class EnvironmentBase:
    def __init__(self, data, device, stride, batch_size, window_size, transaction_cost, initial_capital):
        """
        This class is the environment that interacts with the agent.
        @param data: this is the data_train or data_test in the DataLoader
        those actions) column in the original data-frame of the input data.
        @param device: cpu or gpu
        @param stride: number of steps the environment will take per call to .step()
        @param batch_size:
        @param window_size: for sequential input, the start index for reward is not 0. Therefore, it should be
        provided as a function of window-size.
        @param transaction_cost: cost of each transaction applied in the reward function.
        """
        self.data: pd.DataFrame = data
        self.states: Tensor = None
        self.last_state = 0
        self.window_size = window_size
        self.current_state = 0

        # Flag stating if the agent has its capital invested in the market as shares of the given stock
        self.owns_shares = False
        self.batch_size = batch_size
        self.device = device
        self.stride = stride
        self.close_price = torch.tensor(data.close, dtype=torch.float, device=device)
        self.code_to_action = {0: 'buy', 1: 'None', 2: 'sell'}

        self.transaction_cost = transaction_cost
        self.cost_factor = (1 - self.transaction_cost) ** 2
        self.initial_capital = initial_capital

        # Last step in an episode should be defined 0 value. For us this is:
        self.noop_step = (True, torch.tensor([0.0], dtype=torch.float, device=self.device), None)

    def step(self, action_batch: Tensor) -> tuple[bool, Tensor, Optional[Tensor]]:
        """
        Take a time step of the environment and return the resulting vectors.
        :param action_batch: The action taken as a Tensor containing a single int each. (0 -> Buy, 1 -> None, 2 -> Sell)
        :return: (
            done: A flag representing whether the last step was reached,
            reward: Single float Tensor containing the reward for reaching the current state by taking the provided action,
            next_state: The state reached after taken the provided action as a Tensor
        )
        """
        next_state_idx_0 = self.current_state + self.batch_size
        next_state_idx_n = next_state_idx_0 + self.batch_size

        if next_state_idx_n >= self.last_state:
            # Reached last state in episode
            return self.noop_step
        else:
            return False, self.get_rewards(action_batch), self.states[next_state_idx_0: next_state_idx_n]

    def get_rewards(self, actions: Tensor) -> Tensor:
        """
        Calculate a reward based on the action taken by the agent.
        @param actions: based on the action taken it returns the reward (0 -> Buy, 1 -> None, 2 -> Sell)
        @return: reward
        """
        #print(actions, self.current_state, self.close_price.size(dim=0))
        # First value the reward function knows to calculate the reward
        reward_candle_0 = self.current_state

        # Last value the reward function knows to calculate the reward
        nth_future_candle = self.stride if self.current_state + self.stride < self.last_state else self.close_price.size(dim=0) - 1
        reward_candle_1 = reward_candle_0 + nth_future_candle

        cp0: Tensor = self.close_price[reward_candle_0: reward_candle_0 + self.batch_size]
        cp1: Tensor = self.close_price[reward_candle_1: reward_candle_1 + self.batch_size]
        cp0 = torch.unsqueeze(cp0, 1)
        cp1 = torch.unsqueeze(cp1, 1)

        # This is probably slow, precalculate when the agent holds shares in the stock:
        #print(f"ACTIONS: {actions}")
        share_tensor = torch.zeros_like(actions, device=self.device)
        for i, a in enumerate(actions):
            share_tensor[i] = 1 if self.owns_shares else 0
            if a == 0:
                self.owns_shares = True
            elif a == 2:
                self.owns_shares = False

        # Calculate the reward for the whole batch at once using torch
        rewards = torch.zeros_like(cp0)
        condition_1 = ((actions == 0) | ((actions == 1) & (share_tensor == 1)))
        rewards[condition_1] = torch.mul(
            torch.sub(torch.mul(self.cost_factor, torch.div(cp1[condition_1], cp0[condition_1])), 1), 100)

        condition_2 = ((actions == 2) | ((actions == 1) & (share_tensor == 0)))
        rewards[condition_2] = torch.mul(
            torch.sub(torch.mul(self.cost_factor, torch.div(cp0[condition_2], cp1[condition_2])), 1), 100)

        """
        # OLD CODE THAT DOES NOT USE TENSOR OPERATIONS BUT ACHIEVE THE SAME THING:
        
        if actions == 0 or (actions == 1 and self.owns_shares):
            # profit the stock made in percent, constrained by transaction costs
            # (this might not be accurate due to the transaction costs only effecting the profit of the stock itself)
            reward = (cost_factor * cp1 / cp0 - 1) * 100

        # The agent wants to sell its shares or does not hold any shares at the moment and wants to do nothing
        elif actions == 2 or (actions == 1 and not self.owns_shares):
            # profit the stock lost in percent, constrained by transaction costs
            # This is the same function as above but the quotient is inverted
            reward = (cost_factor * cp0 / cp1 - 1) * 100
        """
        #print(rewards)
        return rewards

    def __iter__(self):
        self.current_state = 0
        self.owns_shares = False
        return self

    def __next__(self):
        if self.current_state < self.last_state:
            batch = self.states[self.current_state:self.current_state + self.batch_size]
            self.current_state += self.batch_size
            return batch
        raise StopIteration

