from typing import Tuple

import torch
from pandas import DataFrame, Series
from torch import Tensor
from torch.types import Device

from agent.algorithm.environment import Environment


def get_share_holding_tensor(actions: Tensor, env: Environment) -> Tensor:
    # This is probably slow, precalculate when the agent holds shares in the stock:
    share_tensor = torch.zeros_like(actions, device=env.device)
    for i, a in enumerate(actions):
        share_tensor[i] = 1 if env.owns_shares else 0
        if a == 0:
            env.owns_shares = True
        elif a == 2:
            env.owns_shares = False
    return share_tensor


def latest_batch(env: Environment) -> Tuple[float, float]:
    """Returns the data indices corresponding to the action batch presented."""
    return env.current_state - env.batch_size, env.current_state

def monetary_changes(
        share_holdings: Tensor,
        transaction_costs: float,
        initial_value: float,
        stock_values: DataFrame
) -> Tuple[Tensor, float]:
    stock_values = stock_values.reset_index(drop=True)
    stock_values['is_invested'] = share_holdings.detach().cpu().numpy()
    stock_values['was_invested'] = stock_values['is_invested'].shift(periods=1, fill_value=stock_values['is_invested'][0])
    stock_values['last_close'] = stock_values['close'].shift(periods=1, fill_value=stock_values['open'][0])
    capital = []
    capital_change = []
    for i, row in stock_values.iterrows():
        if i == 0:
            capital.append(initial_value)
            capital_change.append(0)
        else:
            if row['is_invested'] == 1 and row['was_invested'] == 1:
                stock_delta = 1 + (row['close'] - row['last_close']) / row['last_close']
                capital.append(capital[-1] * stock_delta)

            elif row['is_invested'] == 1 and row['was_invested'] == 0:
                stock_delta = 1 + (row['close'] - row['open']) / row['open']
                capital.append(capital[-1] * stock_delta * (1 - transaction_costs))

            elif row['is_invested'] == 0 and row['was_invested'] == 1:
                stock_delta = 1 + (row['open'] - row['last_close']) / row['last_close']
                capital.append(capital[-1] * stock_delta * (1 - transaction_costs))
            else:  # 0 and 0
                capital.append(capital[-1])
            capital_change.append(capital[-1] - capital[-2])

    return torch.tensor(capital_change, device="cuda:0").unsqueeze(1), capital[-1]


def sum_up_all_in(
        initial_value: float,
        stock_values: DataFrame,
        device: torch.device
) -> Tensor:
    share_holdings = torch.ones(stock_values.shape[0], device=device).unsqueeze(1)
    return sum_up_monetary_value(share_holdings, 0.0, initial_value, stock_values)


def sum_up_monetary_value(
        share_holdings: Tensor,
        transaction_costs: float,
        initial_value: float,
        stock_values: DataFrame
) -> Tensor:

    # ToDo: Scaling here seems off - i need to double check this function !!!

    stock_values = stock_values.reset_index(drop=True)
    stock_values['is_invested'] = share_holdings.detach().cpu().numpy()
    stock_values['was_invested'] = stock_values['is_invested'].shift(periods=1, fill_value=stock_values['is_invested'][0])
    stock_values['last_close'] = stock_values['close'].shift(periods=1, fill_value=stock_values['open'][0])
    capital = []
    for i, row in stock_values.iterrows():
        if i == 0:
            capital.append(initial_value)
        else:
            if row['is_invested'] == 1 and row['was_invested'] == 1:
                stock_delta = 1 + (row['close'] - row['last_close']) / row['last_close']
                capital.append(capital[-1] * stock_delta)

            elif row['is_invested'] == 1 and row['was_invested'] == 0:
                stock_delta = 1 + (row['close'] - row['open']) / row['open']
                capital.append(capital[-1] * stock_delta * (1 - transaction_costs))

            elif row['is_invested'] == 0 and row['was_invested'] == 1:
                stock_delta = 1 + (row['open'] - row['last_close']) / row['last_close']
                capital.append(capital[-1] * stock_delta * (1 - transaction_costs))
            else:  # 0 and 0
                capital.append(capital[-1])

    return torch.tensor(capital, device="cuda:0").unsqueeze(1)


def sum_up_monetary_value_torch(
        share_holdings: Tensor,
        transaction_costs: float,
        initial_value: float,
        stock_values: DataFrame,
        device: Device
) -> float:

    stock_values = stock_values.reset_index(drop=True)
    is_invested_t = share_holdings
    ## .shift(periods=1, fill_value=stock_values['open'][0])
    last_close_t = torch.tensor(stock_values['close'].numpy(),device=device)
    last_close_t.roll(shifts=1, dims=0)
    print(last_close_t)
    ref_0 = []
    ref_1 = []
    for i, row in stock_values.iterrows():
        if row['is_invested'] == 1:
            ref_0.append(row['open'])
            ref_1.append(row['close'] * (1 - transaction_costs))
        elif row['is_invested'] == 0 and stock_values.iloc[i-1]['is_invested'] == 1:
            ref_0.append(row['last_close'])
            ref_1.append(row['open'] * (1 - transaction_costs))
        else:
            ref_0.append(1)
            ref_1.append(1)

    stock_values['ref_0'] = Series(ref_0)
    stock_values['ref_1'] = Series(ref_1)
    stock_values['delta'] = (stock_values['ref_1'] - stock_values['ref_0']) / stock_values['ref_0']

    # DOuble and tripple check this
    perc_change = 0.0
    for i, row in stock_values.iterrows():
        c_1 = row['delta']
        perc_change = perc_change + c_1 + (perc_change * c_1)

    return (1 + perc_change) * initial_value

def monetary_changes_torch(
        share_holdings: Tensor,
        transaction_costs: float,
        initial_value: float,
        stock_values: DataFrame,
        device: Device
) -> Tuple[Tensor, float]:
    """CUDA implementation of above function. This currently is slower than the "less optimized" version"""

    stock_values = stock_values.reset_index(drop=True)

    was_invested = torch.roll(share_holdings, shifts=1)
    was_invested[0] = share_holdings[0]
    #print(is_invested, was_invested)
    #print("-"*80)
    open_t = torch.tensor(stock_values['open'].values, device=device).unsqueeze(1)
    close = torch.tensor(stock_values['close'].values, device=device).unsqueeze(1)
    last_close = torch.roll(close, shifts=1)
    last_close[0] = stock_values['open'][0]
    #print(close, last_close)

    # Building up a tensor of change factors
    factors = torch.ones_like(close)

    invested = ((share_holdings == 1) & (was_invested == 1))
    bought = ((share_holdings == 1) & (was_invested == 0))
    sold = ((share_holdings == 0) & (was_invested == 1))

    factors[invested] = 1 + (close[invested] - last_close[invested]) / last_close[invested]
    factors[bought] = (1 + (close[bought] - open_t[bought]) / open_t[bought]) * (1 - transaction_costs)
    factors[sold] = (1 + (open_t[sold] - last_close[sold]) / last_close[sold]) * (1 - transaction_costs)

    # Calculating the timeseries data // I dont know (yet) how to do this fast using cuda
    capital = torch.zeros_like(close)
    capital_change = torch.zeros_like(close)
    for i, fac in enumerate(factors):
        if i == 0:
            capital[i] = initial_value
        else:
            capital[i] = capital[i-1] * fac
            capital_change[i] = capital[i] - capital[i-1]

    return capital_change, capital[-1]
