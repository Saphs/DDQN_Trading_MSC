from typing import List

import numpy as np
import torch
from numpy import ndarray
from pandas import DataFrame

from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.environment_base import EnvironmentBase


def _flatten_float(_l: List[List[float]]):
    return [item for sublist in _l for item in sublist]


class Environment(EnvironmentBase):

    @classmethod
    def init(cls, data: DataFrame, config: DqnConfig, device: torch.device = 'cpu'):
        return Environment(
            data,
            config.observation_space,
            device,
            stride=config.environment.stride,
            batch_size=config.batch_size,
            window_size=config.window_size,
            transaction_cost=config.environment.transaction_cost,
            initial_capital=config.environment.initial_capital
        )

    def __init__(self,
                 data: DataFrame,
                 state_mode: int,
                 device: torch.device = 'cpu',
                 stride: int = 4,
                 batch_size: int = 50,
                 window_size: int = 1,
                 transaction_cost: float = 0.0,
                 initial_capital: float = 0.0
    ):
        """
        :@param state_mode
                = 1 for OHLC
                = 2 for OHLC + trend
                = 3 for OHLC + trend + %body + %upper-shadow + %lower-shadow
                = 4 for %body + %upper-shadow + %lower-shadow
                = 5 a window of k candles + the trend of the candles inside the window
        :@param action_name
            Name of the column of the action which will be added to the data-frame of data after finding the strategy by
            a specific model.
        :@param device
            GPU or CPU selected by pytorch
        @param stride: number of steps to take each time .step() is called.
        @param batch_size: create batches of observations of size batch_size
        @param window_size: the number of sequential candles that are selected to be in one observation
        @param transaction_cost: cost of the transaction which is applied in the reward function.
        """
        super().__init__(
            data,
            device,
            stride,
            batch_size,
            transaction_cost=transaction_cost,
            initial_capital=initial_capital
        )

        self._mode_map = {
            1: ['open_norm', 'high_norm', 'low_norm', 'close_norm'],
            2: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'trend'],
            3: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'trend', '%body', '%upper-shadow', '%lower-shadow'],
            4: ['%body', '%upper-shadow', '%lower-shadow'],
            5: ['open_norm', 'high_norm', 'low_norm', 'close_norm']
        }

        self.window_size = window_size
        self.state_mode = state_mode
        if state_mode <= 5:
            data_columns = self._mode_map[state_mode]
            self.state_size = window_size * len(data_columns)
            candle_window = []

            temp = None
            for i, row in self.data[data_columns].reset_index(drop=True).iterrows():
                candle_window.append(row.values.copy())
                if i >= window_size - 1:
                    if temp is None:
                        temp = np.array(_flatten_float(candle_window))
                        temp = np.expand_dims(temp, axis=0)
                    else:
                        temp = np.append(temp, np.array([_flatten_float(candle_window)]), axis=0)
                    candle_window = candle_window[1:]
            self.states = torch.tensor(temp, dtype=torch.float, device=device, requires_grad=False)
            #print(self.states[:5])
            self.last_state = temp.shape[0]

        else:
            raise ValueError(f"Unknown state definition: {self.state_mode=}. Supported are: {self._mode_map}")
