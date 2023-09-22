import hashlib
import math
from typing import Callable

import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch import Tensor

from agent.algorithm import reward_tooling as rt
from agent.algorithm.environment import RewardFunction, Environment

def _hash(val):
    return hashlib.md5(f"{val}".encode("UTF-8")).hexdigest()

def _original_work(actions: Tensor, env: Environment) -> Tensor:
    if 'current_capital' not in env.dyn_context:
        env.dyn_context['current_capital'] = env.initial_capital

    # The state has already advanced one step - we need to take that back for this calculation
    reward_candles_0 = env.current_state - env.step_size
    reward_candles_1 = reward_candles_0 + env.look_ahead
    cp0: Tensor = env.close_prices[reward_candles_0: reward_candles_0 + env.batch_size]
    cp0 = torch.unsqueeze(cp0, 1)

    if reward_candles_1 + env.batch_size < env.last_state-1:
        cp1: Tensor = env.close_prices[reward_candles_1: reward_candles_1 + env.batch_size].unsqueeze(1)
    else:
        # If we are reaching the last value and the batch is not full we pad with the
        # last value to make bulk operations work.
        cp1: Tensor = env.close_prices[reward_candles_1: env.last_state-1].unsqueeze(1)
        padding_size = cp0.size(0) - cp1.size(0)
        last_val = env.close_prices[env.last_state-1]
        cp1 = torch.nn.functional.pad(cp1, (0, 0, 0, padding_size), 'constant', value=last_val)

    # This is probably slow, precalculate when the agent holds shares in the stock:
    share_tensor = rt.get_share_holding_tensor(actions, env)

    # Calculate the reward for the whole batch at once using torch
    rewards = torch.zeros_like(cp0, device=env.device)
    condition_1 = ((actions == 0) | ((actions == 1) & (share_tensor == 1)))
    rewards[condition_1] = (env.cost_factor * (cp1[condition_1] / cp0[condition_1]) - 1) * 100
    condition_2 = ((actions == 2) | ((actions == 1) & (share_tensor == 0)))
    rewards[condition_2] = (env.cost_factor * (cp0[condition_2] / cp1[condition_2]) - 1) * 100

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
    return rewards


def _simple_profit_function(actions: Tensor, env: Environment) -> Tensor:
    if 'current_capital' not in env.dyn_context:
        env.dyn_context['current_capital'] = env.initial_capital

    previous_capital = env.dyn_context['current_capital']
    # Calculating batch position that lead to the give actions.
    b_start, b_end = rt.latest_batch(env)
    stock_values = env.base_data[b_start:b_end][['open', 'close']]  # ToDo: Make this a tensor operation

    # This function has side effects and sets the "env.owns_shares" value in the environment.
    share_holdings = rt.get_share_holding_tensor(actions, env)
    reward_t, capital = rt.monetary_changes(
        share_holdings=share_holdings,
        transaction_costs=env.transaction_cost,
        initial_value=previous_capital,
        stock_values=stock_values,
    )
    env.dyn_context['current_capital'] = capital
    return reward_t


def _discrete_rewards(actions: Tensor, env: Environment) -> Tensor:
    env.dyn_context['current_capital'] = 0

    # Calculating batch position that lead to the give actions.
    b_start, b_end = rt.latest_batch(env)
    stock_values = env.base_data[b_start:b_end+2][['open', 'close']]  # ToDo: Make this a tensor operation
    stock_values['open_p3'] = stock_values['open'].shift(periods=-2)
    stock_values = stock_values.dropna()
    stock_values['trajectory'] = stock_values['open_p3'] - stock_values['open']

    traj_t = torch.tensor(stock_values['trajectory'].values, device=env.device).unsqueeze(1)
    positive = (((actions == 0) & (traj_t > 0.25)) | ((actions == 2) & (traj_t < 0.25)))
    rewards = torch.clone(actions).fill_(-10.0).type(torch.float)
    rewards[positive] = 5.0
    return rewards


def _delta_to_BH(actions: Tensor, env: Environment) -> Tensor:

    if 'current_capital' not in env.dyn_context:
        env.dyn_context['current_capital'] = env.initial_capital
    previous_capital = env.dyn_context['current_capital']
    if 'buy_n_hold' not in env.dyn_context:
        stock_values = env.base_data[['open', 'close']]
        buy_hold_t = rt.sum_up_all_in(
            initial_value=previous_capital,
            stock_values=stock_values,
            device=env.device
        )
        env.dyn_context['buy_n_hold'] = buy_hold_t
    buy_hold_t = env.dyn_context['buy_n_hold']

    b_start, b_end = rt.latest_batch(env)
    stock_values = env.base_data[b_start:b_end]
    share_holdings = rt.get_share_holding_tensor(actions, env)
    capital_t = rt.sum_up_monetary_value(share_holdings, env.transaction_cost, previous_capital, stock_values)

    # highly experimental value that tries to give an incentive to trade
    """
    t_buy = actions[actions == 0].size()[0]
    t_sell = actions[actions == 2].size()[0]
    variety = actions.size()[0] - math.fabs(t_buy - t_sell)
    if actions[actions == 0].size()[0] >= actions.size()[0] - 2:
        variety2 = -50
    elif actions[actions == 2].size()[0] >= actions.size()[0] - 2:
        variety2 = -50
    else:
        variety2 = 10
    """


    delta = (capital_t - buy_hold_t[b_start:b_end]) # + variety + variety2
    #data_frame = DataFrame({"buy_n_hold": buy_hold_t.view(-1).tolist()[b_start:b_end], "capital": capital_t.view(-1).tolist(), "delta": delta.view(-1).tolist()})
    env.dyn_context['current_capital'] = capital_t[-1].item()
    return delta


class RewardFunctionProvider:

    function_map = {
        "original_work": _original_work,
        "simple_profit": _simple_profit_function,
        "discrete_rewards": _discrete_rewards,
        "delta_to_bh": _delta_to_BH
    }

    @classmethod
    def get(cls, name: str) -> RewardFunction:
        return cls.function_map.get(name)



