import hashlib
import logging
import math
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import ast
import torch
from pandas import DataFrame
from torch import Tensor

from agent.algorithm.config_parsing.dqn_config import DqnConfig
from data_access.data_loader import _add_normalized_data
from data_access.pattern_detection.label_candles import label_candles


def _flatten_float(_l: List[List[float]]):
    return [item for sublist in _l for item in sublist]


# "ObservationSpace" parameter to actual observation space column mapping
_MODE_MAP = {
    1: ['open_norm', 'high_norm', 'low_norm', 'close_norm'],
    2: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'trend'],
    3: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'trend', '%body', '%upper-shadow', '%lower-shadow'],
    4: ['%body', '%upper-shadow', '%lower-shadow'],
    5: ['open', 'high', 'low', 'close']
}


class Environment:
    """Environment a given agent is trained in. For us this is a simulated stock market"""

    def __init__(self, data: DataFrame, config: DqnConfig, device: torch.device = 'cpu'):
        # Constants kept for either bookkeeping or performance reasons
        self.device = device
        self.transaction_cost = config.environment.transaction_cost
        self.cost_factor = (1 - self.transaction_cost) ** 2
        self.initial_capital = config.environment.initial_capital
        self.batch_size = config.batch_size
        self.step_size = config.environment.step_size
        self.look_ahead = config.environment.look_ahead
        self.density = config.environment.density
        self.window_size = config.window_size
        self.state_mode = config.observation_space
        # Last step in an episode should be defined 0 value. For us this is:
        self.noop_step = (True, torch.tensor([0.0], dtype=torch.float, device=self.device), None)

        # Dynamic state values that depend on the data or change in respect to time
        self.base_data: pd.DataFrame = data
        # ToDo: Is that the secret sauce??
        #self.base_data.ffill(inplace=True)
        self.close_prices = torch.tensor(data.close, dtype=torch.float, device=device)
        self.states: Tensor = None
        self.last_state = 0
        self.current_state = 0
        # Context object allowed to be dynamically changed by reward functions to store state values
        self.dyn_context = {}
        # Flag stating if the agent has its capital invested in the market as shares of the given stock
        self.owns_shares = False
        self.reward_function = RewardFunctionProvider.get(config.reward_function)

        # Calculate actual states that will be presented to a potential agent later
        self._prepare_states()

    def __iter__(self):
        self.current_state = 0
        self.owns_shares = False
        self.dyn_context = {}
        return self

    def __next__(self):
        # ToDo: There is no case for the rest of the data
        #  - the last batch should be a partial batch and need special treatment
        if self.current_state < self.last_state:
            batch = self.states[self.current_state:self.current_state + self.batch_size]
            self.current_state += self.step_size
            return batch
        raise StopIteration

    def __len__(self):
        return math.ceil(self.last_state / self.batch_size)

    def _prepare_states(self):
        if self.state_mode <= 5:
            data_columns: list[str] = _MODE_MAP[self.state_mode]
            self.state_size = self.window_size * len(data_columns)

            self.base_data = self._apply_density()

            # Expand the data to accommodate for window shift
            state_columns: list[str] = data_columns.copy()
            for i in range(1, self.window_size):
                shift_df = self.base_data[data_columns].shift(periods=i)
                shift_df = shift_df.add_prefix(f'{i:04}_')
                state_columns += shift_df.keys().values.tolist()
                self.base_data = pd.concat([self.base_data, shift_df], axis=1)

            # Drop first n = window size - 1 rows as they now contains NaN
            self.base_data = self.base_data.dropna(subset=state_columns).copy()
            self.base_data['state'] = self.base_data[state_columns].apply(lambda r: np.array(r), axis=1)

            #print(self.base_data['state'])
            data_array = np.stack(self.base_data.reset_index(drop=True)['state'], axis=0)
            self.states = torch.tensor(data_array, dtype=torch.float, device=self.device, requires_grad=False)

            # Save some constants for later
            self.starting_offset = self.window_size - 1
            self.last_state = self.states.shape[0]
        else:
            raise ValueError(f"Unknown state definition: {self.state_mode=}. Supported are: {_MODE_MAP}")

    def _apply_density(self) -> DataFrame:

        if self.density == 1:
            logging.info(f"Density set to 1 no aggregation is taking place.")
            return self.base_data

        df = self.base_data.copy()

        # Define aggregations
        def _last(series):
            return series.iloc[-1]

        def _first(series):
            return series.iloc[0]

        agg_dict = {
            'date': _last,
            'open': _first,
            'high': 'max',
            'low': 'min',
            'close': _last,
        }

        logging.info(f"Density >1 aggregating every {self.density} rows into one.")
        df.reset_index(inplace=True)
        df.index = df.index.astype(int)
        df = df.reset_index().groupby(df.index // self.density).agg(agg_dict)
        logging.info("Relabeling data after aggregate.")
        label_candles(df)
        df = _add_normalized_data(df)
        return df

    def act(self, action_batch: Tensor) -> tuple[bool, Tensor, Optional[Tensor]]:
        """
        Take a time step of the environment and return the resulting vectors.
        :param action_batch: The action taken as a Tensor containing a single int each. (0 -> Buy, 1 -> None, 2 -> Sell)
        :return: (
            done: A flag representing whether the last step was reached,
            reward: Single float Tensor containing the reward for reaching the current state by taking the provided action,
            next_state: The state reached after taken the provided action as a Tensor
        )
        """
        # current state is already updated from the iteration step
        next_state_idx_0 = self.current_state
        next_state_idx_n = next_state_idx_0 + self.batch_size
        #print(f"\n{self.current_state=}, {next_state_idx_0=}, {next_state_idx_n=}, {self.batch_size=}, {self.last_state=}")
        #print(self.states[next_state_idx_0: next_state_idx_n])
        if next_state_idx_n >= self.last_state:
            # Reached last state in episode
            return self.noop_step
        else:
            return False, self.reward_function(action_batch, self), self.states[next_state_idx_0: next_state_idx_n]


# ToDo: Find a way around this wierd import after the class,
#  this is needed as importing it earlier causes a circular dependency
RewardFunction = Callable[[Tensor, Environment], Tensor]
from agent.algorithm.reward_function_provider import RewardFunctionProvider
