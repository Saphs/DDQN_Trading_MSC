from typing import Callable

import torch
from torch import Tensor

from agent.algorithm import reward_tooling as rt
from agent.algorithm.environment import RewardFunction, Environment


def _func_zero(actions: Tensor, env: Environment) -> Tensor:
    reward_candle_0 = env.current_state

    # Last value the reward function knows to calculate the reward
    nth_future_candle = env.stride if env.current_state + env.stride < env.last_state else env.close_prices.size(
        dim=0) - 1
    reward_candle_1 = reward_candle_0 + nth_future_candle

    cp0: Tensor = env.close_prices[reward_candle_0: reward_candle_0 + env.batch_size]
    cp1: Tensor = env.close_prices[reward_candle_1: reward_candle_1 + env.batch_size]
    cp0 = torch.unsqueeze(cp0, 1)
    cp1 = torch.unsqueeze(cp1, 1)

    # This is probably slow, precalculate when the agent holds shares in the stock:
    # print(f"ACTIONS: {actions}")
    share_tensor = torch.zeros_like(actions, device=env.device)
    for i, a in enumerate(actions):
        share_tensor[i] = 1 if env.owns_shares else 0
        if a == 0:
            env.owns_shares = True
        elif a == 2:
            env.owns_shares = False

    # Calculate the reward for the whole batch at once using torch
    rewards = torch.zeros_like(cp0)
    condition_1 = ((actions == 0) | ((actions == 1) & (share_tensor == 1)))
    rewards[condition_1] = torch.mul(
        torch.sub(torch.mul(env.cost_factor, torch.div(cp1[condition_1], cp0[condition_1])), 1), 100)

    condition_2 = ((actions == 2) | ((actions == 1) & (share_tensor == 0)))
    rewards[condition_2] = torch.mul(
        torch.sub(torch.mul(env.cost_factor, torch.div(cp0[condition_2], cp1[condition_2])), 1), 100)

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

class RewardFunctionProvider:

    function_map = {
        "func_zero": _func_zero,
        "simple_profit": _simple_profit_function
    }

    @classmethod
    def get(cls, name: str) -> RewardFunction:
        return cls.function_map.get(name)



