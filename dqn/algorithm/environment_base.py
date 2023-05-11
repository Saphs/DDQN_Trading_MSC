from typing import List, Tuple

import pandas as pd
import torch
import math


class EnvironmentBase:
    def __init__(self, data, action_name, device, n_step, batch_size, start_index_reward, transaction_cost):
        """
        This class is the environment that interacts with the agent.
        @param data: this is the data_train or data_test in the DataLoader
        @param action_name: This is the name of the action (typically the name of the model who generated
        those actions) column in the original data-frame of the input data.
        @param device: cpu or gpu
        @param n_step: number of future steps the reward function will take into a count
        @param batch_size:
        @param start_index_reward: for sequential input, the start index for reward is not 0. Therefore, it should be
        provided as a function of window-size.
        @param transaction_cost: cost of each transaction applied in the reward function.
        """
        self.data: pd.DataFrame = data
        self.states = []
        self.current_state = -1

        # Flag stating if the agent has its capital invested in the market as shares of the given stock
        self.owns_shares = False
        self.batch_size = batch_size
        self.device = device
        self.n_step = n_step
        self.close_price = list(data.close)
        self.action_name = action_name
        self.code_to_action = {0: 'buy', 1: 'None', 2: 'sell'}

        # Offset the current state starts at. This can be needed when there are n (window size many) steps to skip in the beginning
        # If the input mode does not include multiple candle sticks it should be set to 0
        self.starting_offset = start_index_reward

        self.transaction_cost = transaction_cost

    def get_current_state(self):
        """
        @return: returns current state of the environment
        """
        self.current_state += 1
        if self.current_state == len(self.states):
            return None
        return self.states[self.current_state]

    def step(self, action) -> Tuple[bool, float, List[float]]:
        """
        right now, the reward is one step...
        TODO: I should promote it to n-step reward with baseline (average reward)
        :param action:  0 -> Buy
                        1 -> None
                        2 -> Sell
        :return:
        """
        done = False
        next_state = None

        if self.current_state + self.n_step < len(self.states):
            next_state = self.states[self.current_state + self.n_step]
        else:
            done = True

        if action == 0:
            self.owns_shares = True
        elif action == 2:
            self.owns_shares = False

        reward = 0.0
        if not done:
            reward = self.get_reward(action)

        return done, reward, next_state

    def get_reward(self, action) -> float:
        """
        Calculate a reward based on the action taken by the agent.
        @param action: based on the action taken it returns the reward (0 -> Buy, 1 -> None, 2 -> Sell)
        @return: reward
        """

        # First value the reward function knows to calculate the reward
        reward_candle_0 = self.starting_offset + self.current_state

        # Last value the reward function knows to calculate the reward
        nth_future_candle = self.n_step if self.current_state + self.n_step < len(self.states) else len(self.close_price) - 1
        reward_candle_1 = reward_candle_0 + nth_future_candle

        cp0 = self.close_price[reward_candle_0]
        cp1 = self.close_price[reward_candle_1]

        reward = 0
        cost_factor = (1 - self.transaction_cost) ** 2

        # The agent wants to buy shares or already holds shares at the moment and wants to do nothing
        if action == 0 or (action == 1 and self.owns_shares):
            # profit the stock made in percent, constrained by transaction costs
            # (this might not be accurate due to the transaction costs only effecting the profit of the stock itself)
            reward = (cost_factor * cp1 / cp0 - 1) * 100

        # The agent wants to sell its shares or does not hold any shares at the moment and wants to do nothing
        elif action == 2 or (action == 1 and not self.owns_shares):
            # profit the stock lost in percent, constrained by transaction costs
            # This is the same function as above but the quotient is inverted
            reward = (cost_factor * cp0 / cp1 - 1) * 100

        return reward

    def calculate_reward_for_one_step(self, action, index, rewards):
        """
        The reward for selling is the opposite of the reward for buying, meaning that if some one sells his share and the
        value of the share increases, thus he should be punished. In addition, if some one sells appropriately and the value
        of the share decreases, he should be awarded
        :param action:
        :param index:
        :param rewards:
        :param own_share: whether the user holds the share or not.
        :return:
        """
        index += self.starting_offset  # Last element inside the window
        if action == 0 or (action == 1 and self.owns_shares):  # Buy Share or Hold Share
            difference = self.close_price[index + 1] - self.close_price[index]
            rewards.append(difference)

        elif action == 2 or (action == 1 and not self.owns_shares):  # Sell Share or No Share
            difference = self.close_price[index] - self.close_price[index + 1]
            rewards.append(difference)

    def reset(self):
        self.current_state = -1
        self.owns_shares = False

    def __iter__(self):
        self.index_batch = 0
        self.num_batch = math.ceil(len(self.states) / self.batch_size)
        return self

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [torch.tensor([s], dtype=torch.float, device=self.device) for s in
                     self.states[self.index_batch * self.batch_size: (self.index_batch + 1) * self.batch_size]]
            self.index_batch += 1
            return torch.cat(batch)

        raise StopIteration

    def get_total_reward(self, action_list):
        """
        You should call reset before calling this function, then it receives action batch
        from the input and calculate rewards.
        :param action_list:
        :return:
        """
        total_reward = 0
        for a in action_list:
            if a == 0:
                self.owns_shares = True
            elif a == 2:
                self.owns_shares = False
            self.current_state += 1
            total_reward += self.get_reward(a)

        return total_reward

    def make_investment(self, action_list: List[int]):
        """
        Provided a list of actions at each time-step, it converts the action to its original name like:
        0 -> Buy
        1 -> None
        2 -> Sell
        """
        def _to_action(x: int) -> str:
            return self.code_to_action[x]
        self.data[self.action_name] = list(map(_to_action, action_list))
