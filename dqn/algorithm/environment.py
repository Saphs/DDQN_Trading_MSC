from typing import List

import numpy as np
import torch
from pandas import DataFrame

from dqn.algorithm.environment_base import EnvironmentBase


def _flatten_float(_l: List[List[float]]):
    return [item for sublist in _l for item in sublist]


class Environment(EnvironmentBase):

    def __init__(self,
                 data: DataFrame,
                 state_mode: int,
                 action_name: str,
                 device: torch.device = 'cpu',
                 n_step: int = 4,
                 batch_size: int = 50,
                 window_size: int = 1,
                 transaction_cost: float = 0.0
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
        @param n_step: number of steps in the future to get reward.
        @param batch_size: create batches of observations of size batch_size
        @param window_size: the number of sequential candles that are selected to be in one observation
        @param transaction_cost: cost of the transaction which is applied in the reward function.
        """
        start_index_reward = 0 if state_mode != 5 else window_size - 1
        super().__init__(
            data,
            action_name,
            device,
            n_step,
            batch_size,
            start_index_reward=start_index_reward,
            transaction_cost=transaction_cost
        )

        self._mode_map = {
            1: ['open_norm', 'high_norm', 'low_norm', 'close_norm'],
            2: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'trend'],
            3: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'trend', '%body', '%upper-shadow', '%lower-shadow'],
            4: ['%body', '%upper-shadow', '%lower-shadow'],
            5: ['open_norm', 'high_norm', 'low_norm', 'close_norm']
        }

        self.state_mode = state_mode
        if state_mode < 5:
            data_columns = self._mode_map[state_mode]
            self.state_size = len(data_columns)
            self.data_preprocessed = data[data_columns].values
            for i in range(len(self.data_preprocessed)):
                self.states.append(self.data_preprocessed[i])

        elif state_mode == 5:  # Windowed OHLC mode
            data_columns = self._mode_map[state_mode]
            self.state_size = window_size * len(data_columns)
            candle_window = []

            for i, row in self.data[data_columns].reset_index(drop=True).iterrows():
                candle_window.append(row.values.copy())
                if i >= window_size - 1:
                    self.states.append(np.array(_flatten_float(candle_window)))
                    candle_window = candle_window[1:]

        else:
            raise ValueError(f"Unknown state definition: {self.state_mode=}. Supported are: {self._mode_map}")
