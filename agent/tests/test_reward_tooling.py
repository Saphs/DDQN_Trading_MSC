import unittest

import torch
from pandas import DataFrame
from torch import Tensor

import testing_utilities
from agent.algorithm import reward_tooling as rt


class TestRewardTooling(unittest.TestCase):

    def test_sum_up_monetary_value_always_in(self):
        """Check if monetary values are summed up correctly if always in market."""
        share_holding: Tensor = torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1]])
        initial_value = 1000
        df: DataFrame = testing_utilities.get_test_stock_data()[['open', 'close']].reset_index(drop=True)
        shortened_df = df[:len(share_holding)]
        x0 = shortened_df['close'][0]
        xn = shortened_df['close'][len(shortened_df)-1]
        exp_change: float = (xn - x0) / x0
        calculated_sum = rt.sum_up_monetary_value(share_holding, 0.0, initial_value, shortened_df)

        self.assertAlmostEqual(x0 * (1 + exp_change), 293.838837)  # Sanity check
        self.assertAlmostEqual(initial_value * (1 + exp_change), calculated_sum[-1][0].item())

    def test_sum_up_monetary_value_compared_to_bh(self):
        """Check if Buy and Hold can be recreated using this."""
        df: DataFrame = testing_utilities.get_test_stock_data()[['open', 'close']].reset_index(drop=True)
        share_holding: Tensor = torch.ones(df.shape[0], 1)

        initial_value = df.loc[0][1]
        calculated_sum = rt.sum_up_monetary_value(share_holding, 0.0, initial_value, df)
        _df = df[['close']]
        _df['calc_close'] = calculated_sum.squeeze(1).detach().cpu()
        print(_df)
        self.assertAlmostEqual(237.972977, calculated_sum[-1][0].item())

    def test_sum_up_monetary_value_never_in(self):
        """Check if monetary values are summed up correctly if never in market."""
        share_holding: Tensor = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [0]])
        initial_value = 1000
        df: DataFrame = testing_utilities.get_test_stock_data()[['open', 'close']].reset_index(drop=True)
        shortened_df = df[:len(share_holding)]

        calculated_sum = rt.sum_up_monetary_value(share_holding, 0.0, initial_value, shortened_df)
        self.assertAlmostEqual(initial_value, calculated_sum[-1][0].item())

    def test_sum_up_monetary_value_halfway_out(self):
        """Check if monetary values are summed up correctly if moving out of market."""
        share_holding: Tensor = torch.tensor([[1], [1], [1], [1], [0], [0], [0], [0]])
        initial_value = 1000
        df: DataFrame = testing_utilities.get_test_stock_data()[['open', 'close']].reset_index(drop=True)
        shortened_df = df[:len(share_holding)]
        x0 = shortened_df['close'][0]
        xn = shortened_df['open'][4]  # rising flank
        exp_change: float = (xn - x0) / x0
        calculated_sum = rt.sum_up_monetary_value(share_holding, 0.0, initial_value, shortened_df)

        #print(shortened_df, xn, exp_change, x0 * (1 + exp_change))
        self.assertAlmostEqual(x0 * (1 + exp_change), 296.296295)  # Sanity check
        self.assertAlmostEqual(initial_value * (1 + exp_change), calculated_sum[-1][0].item())

    def test_sum_up_monetary_value_halfway_in(self):
        """Check if monetary values are summed up correctly if moving out of market."""
        share_holding: Tensor = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
        initial_value = 1000
        df: DataFrame = testing_utilities.get_test_stock_data()[['open', 'close']].reset_index(drop=True)
        shortened_df = df[:len(share_holding)]
        x0 = shortened_df['open'][4]
        xn = shortened_df['close'][len(shortened_df)-1]  # rising flank
        exp_change: float = (xn - x0) / x0
        calculated_sum = rt.sum_up_monetary_value(share_holding, 0.0, initial_value, shortened_df)

        print(shortened_df, xn, exp_change, initial_value * (1 + exp_change))
        self.assertAlmostEqual(x0 * (1 + exp_change), 293.838837)  # Sanity check
        self.assertAlmostEqual(initial_value * (1 + exp_change), calculated_sum[-1][0].item())


