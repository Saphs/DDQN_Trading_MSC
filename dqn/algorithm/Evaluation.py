import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
import scipy.stats as sps

from dqn.algorithm.dqn_agent import DqnAgent
from dqn.algorithm.environment import Environment
from dqn.algorithm.model.neural_network import NeuralNetwork


def _arithmetic_mean(values: pd.Series) -> float:
    """The simplest average of a list of numbers"""
    return values.sum() / len(values)


def _geometric_mean(values: pd.Series):
    """
    The geometric mean averages the distance of factors
    See: https://de.wikipedia.org/wiki/Geometrisches_Mittel
    """
    values += 1
    return ((values.values.prod()) ** (1 / len(values))) - 1

class Evaluation:

    def __init__(self,
                 agent: Optional[DqnAgent],
                 env: Environment,
                 stock_name: str,
                 force_eval_mode: bool = False
                 ):

        self._in_reference_mode = agent is None

        # Take the agents policy network (Q*-Function)
        if self._in_reference_mode:
            net: NeuralNetwork = NeuralNetwork(1)
        else:
            net: NeuralNetwork = NeuralNetwork(env.state_size)
            net.load_state_dict(agent.policy_net.state_dict())
            net.to(env.device)
            if force_eval_mode:
                net.eval()

        # Core values that are to be evaluated
        self.q_function = net
        self.env = env

        # Declare required meta values
        self.capital_t0 = env.initial_capital
        self.transaction_cost = env.transaction_cost
        self.stock_name = stock_name
        self.action_column = 'prediction'
        self.portfolio_column = 'portfolio'
        self.model_name = agent.name if not self._in_reference_mode else "Buy&Hold"
        self.evaluation_mode = not self.q_function.training if not self._in_reference_mode else None
        self.metric_table = DataFrame({
            "model_name": [self.model_name],
            "evaluation_mode": [self.evaluation_mode]
        })

        # Apply Q*-Function to the environment & Calculate the outcome of its actions
        logging.info(f"Evaluating {self.model_name} in {self._mode()}-mode on stock: {self.stock_name}.")
        self.raw_data = self._predict()
        self.raw_data = self._calculate_return()
        self.raw_data = self._calculate_perc_return()
        self.raw_data = self._calculate_effective_transactions()

    def _mode(self) -> str:
        if self.evaluation_mode is None:
            return "Reference"
        else:
            return "Evaluation" if self.evaluation_mode else "Training"

    def _predict(self) -> DataFrame:
        """
        Applies the Q*-function, that should be evaluated to the environment's data.
        :return: DataFrame containing the actions the agent would take to maximize the monetary value of its portfolio.
        """
        with torch.no_grad():
            action_list = []
            if self._in_reference_mode:
                # Build a synthetic action list that only buys at the beginning and then does nothing
                action_list = [1] * self.env.data.shape[0]
                action_list[0] = 0
            else:
                self.env.__iter__()
                for i, batch in enumerate(self.env):
                    try:
                        # net(batch) -> apply network to batch
                        # .max(1)[1] -> get index of max value
                        # .cpu().numpy() -> access on cpu if on cuda and cast to numpy array
                        action_list += list(self.q_function(batch).max(1)[1].cpu().numpy())
                    except ValueError:
                        print("Error")
                        action_list += [1]

            def _to_action(x: int) -> str:
                return self.env.code_to_action[x]

            self.env.data[self.action_column + '_numeric'] = action_list
            self.env.data[self.action_column] = list(map(_to_action, action_list))
            return self.env.data

    def _calculate_return(self) -> DataFrame:
        """
        Calculates the portfolio value at each point in time.
        The function assumes the model trades each morning to the opening price.
        :return: returns an altered dataframe containing the raw data with the portfolio value added to it.
        """

        df = self.raw_data
        capital_values = []
        is_invested = False

        for i, row in df.reset_index().iterrows():
            if i == 0:
                capital_values.append(self.capital_t0)
            else:
                last_decision = df.iloc[i-1][self.action_column]
                last_close = df.iloc[i-1]['close']
                next_capital = capital_values[-1]
                if not is_invested:
                    if last_decision == 'buy':
                        is_invested = True
                        stock_delta = 1 + (row['close'] - row['open']) / row['open']
                        next_capital = next_capital * (1 - self.transaction_cost) * stock_delta
                    elif last_decision == 'None' or last_decision == 'sell':
                        pass
                elif is_invested:
                    if last_decision == 'buy' or last_decision == 'None':
                        stock_delta = 1 + (row['close'] - last_close) / last_close
                        next_capital = next_capital * stock_delta
                    elif last_decision == 'sell':
                        is_invested = False
                        stock_delta = 1 + (row['open'] - last_close) / last_close
                        next_capital = next_capital * (1 - self.transaction_cost) * stock_delta
                capital_values.append(next_capital)

        df[self.portfolio_column] = capital_values
        return df

    def _calculate_perc_return(self) -> DataFrame:
        df = self.raw_data.reset_index()
        percentage_changes = []
        for i, row in df.iterrows():
            if i == 0:
                percentage_changes.append(0.0)
            else:
                last_val = df.iloc[i-1][self.portfolio_column]
                percentage_changes.append((row[self.portfolio_column] - last_val) / last_val)
        df['perc_change'] = percentage_changes
        return df

    def _calculate_effective_transactions(self) -> DataFrame:
        """Mark transactions that actually effected the model's performance in a new column."""
        white_list = ["buy", "sell"]
        df = self.raw_data

        filtered_df = df[df[self.action_column].isin(white_list)]
        filtered_df['last_action'] = filtered_df[self.action_column].shift(periods=1, fill_value='sell')

        eff_transactions = [False] * len(df)
        for i, row in filtered_df.iterrows():
            eff_transactions[i] = row[self.action_column] != row['last_action']

        self.raw_data['effective_transaction'] = eff_transactions
        return self.raw_data

    def calculate_metric_table(self) -> DataFrame:
        self.metric_table['stock'] = self.stock_name
        self.metric_table['V_i'] = round(self.raw_data.iloc[0][self.portfolio_column], 2)
        self.metric_table['V_f'] = round(self.raw_data.iloc[-1][self.portfolio_column], 2)
        self.metric_table['profit'] = self.profit()
        self.metric_table['return'] = self.holding_period_return()
        self.metric_table['amean_return'] = self.amean_return()
        self.metric_table['gmean_return'] = self.gmean_return()
        self.metric_table['rate_of_return'] = self.rate_of_return()
        self.metric_table['log_return'] = self.log_return()
        self.metric_table['log_rate_of_return'] = self.log_rate_of_return()
        self.metric_table['daily_volatility'] = self.daily_volatility()
        self.metric_table['total_volatility'] = self.total_volatility()
        self.metric_table['sharpe_ratio'] = self.sharpe_ratio()
        self.metric_table['value_at_risk_per_day'] = self.value_at_risk_per_day()
        self.metric_table['value_at_risk'] = self.value_at_risk_total()
        self.metric_table['tried_transactions'] = self.tried_transactions()
        self.metric_table['transactions'] = self.transactions()
        self.metric_table['transactions_per_30_t'] = self.transactions_per_t()
        return self.metric_table

    def profit(self) -> float:
        """
        Calculate the profit.
        P = V_f - V_i

        :return Absolute increase of the portfolio after the complete investment period.
        """
        v_i = self.raw_data.iloc[0][self.portfolio_column]
        v_f = self.raw_data.iloc[-1][self.portfolio_column]
        return round(v_f - v_i, 2)

    def holding_period_return(self) -> float:
        """
        Calculate the (holding period) return.
        R = (V_f - V_i) / V_i

        See: https://en.wikipedia.org/wiki/Rate_of_return
        :return Percentage in-/decrease over the complete investment period.
        """
        v_i = self.raw_data.iloc[0][self.portfolio_column]
        v_f = self.raw_data.iloc[-1][self.portfolio_column]
        return (v_f - v_i) / v_i

    def amean_return(self) -> float:
        """Calculate the average return per time step in percent."""
        return _arithmetic_mean(self.raw_data['perc_change'])

    def gmean_return(self) -> float:
        """Calculate the average return per time step in percent."""
        return _geometric_mean(self.raw_data['perc_change'])

    def rate_of_return(self):
        """
        Calculate the rate of return.
        r = (1 + R)^(1/t) - 1
        :return The rate the return increased per time unit, taking compounding effects into a count.
        """
        t = self.raw_data.shape[0]
        _R = self.holding_period_return()
        return (1 + _R) ** (1 / t) - 1

    def log_return(self):
        """
        Calculate the logarithmic return (continuously compounded return).
        R_log = ln( V_f / V_i )
        :return: ??? ToDo: Understand this value
        """
        v_i = self.raw_data.iloc[0][self.portfolio_column]
        v_f = self.raw_data.iloc[-1][self.portfolio_column]
        return math.log(v_f / v_i)

    def log_rate_of_return(self):
        """
        Calculate the logarithmic return.
        r_log = R_log / t
        :return: ??? ToDo: Understand this value
        """
        return self.log_return() / self.raw_data.shape[0]

    def tried_transactions(self) -> int:
        """The number of times the strategy or model tried to buy or sell its stocks."""
        white_list = ["buy", "sell"]
        filtered_df = self.raw_data[self.raw_data[self.action_column].isin(white_list)]
        return filtered_df.shape[0]

    def transactions(self) -> int:
        """The number of times the strategy or model actually bought or sold stocks."""
        return self.raw_data.loc[self.raw_data['effective_transaction']].shape[0]

    def transactions_per_t(self, bulk: int = 30) -> float:
        """
        The number of transactions per 30 time units.
        Assuming the time unit of the original data is representative of a day, this will result in a rough
        estimation of transactions per month.

        Note: This will only take transactions that effected the strategies stocks into a count.
        """
        return round((self.transactions() / self.raw_data.shape[0]) * bulk, 2)

    def daily_volatility(self) -> float:
        """
        Calculate the daily volatility, e.g.: the std. deviation of the log returns.
        :return: The volatility describing the expected changes to the log return per day
        """
        tmp_df = self.raw_data.reset_index()[[self.portfolio_column]]
        log_returns = []
        for i, row in tmp_df.iterrows():
            v_i = tmp_df.iloc[i-1] if not i == 0 else row[self.portfolio_column]
            v_f = row[self.portfolio_column]
            log_returns.append(math.log(v_f / v_i))
        tmp_df['log_return_daily'] = log_returns
        return tmp_df['log_return_daily'].std()

    def total_volatility(self) -> float:
        """
        Calculate the total volatility, e.g.:
        σ_total = σ_daily * sqrt(T)
        where T is the number of time steps in the data set.
        :return: The volatility describing the expected changes to the log return over the whole trading period
        """
        return self.daily_volatility() * np.sqrt(self.raw_data.shape[0])

    def _profit_by(self, interest_rate: float) -> float:
        k_0 = self.raw_data.iloc[0][self.portfolio_column]
        n = math.floor(self.raw_data.shape[0] / 30)
        return k_0 * (1 + interest_rate) ** n

    def _return_by(self, interest_rate: float) -> float:
        v_i = self.raw_data.iloc[0][self.portfolio_column]
        v_f = self._profit_by(interest_rate)
        return (v_f - v_i) / v_i

    def sharpe_ratio(self, risk_free_interest_rate: float = 0.024960):
        """
        Calculate the sharpe-ratio. See: https://de.wikipedia.org/wiki/Sharpe-Quotient
        Note that the default value for this is set to 2.4% as the "Germany Government Bond 10Y" has this value
        and is forecast to rise in 2024. If this is a good choice remains to be validated by Plutos. See:
        https://tradingeconomics.com/germany/government-bond-yield. Its calculated:

        S = (R_total - R_risk-free) / σ_total

        :param risk_free_interest_rate: interest of a known risk-free strategy
        :return: a measure of quality of an investment in comparison to a risk-free investment
        """
        _R_total = self.holding_period_return()
        _R_risk_free = self._return_by(risk_free_interest_rate)
        return (_R_total - _R_risk_free) / self.total_volatility()

    def _value_at_risk(self, return_per_t: float, volatility: float, alpha: float = 0.05):
        confidence = 1 - alpha
        dist = sps.norm(loc=return_per_t, scale=volatility)
        alpha_quantile = dist.ppf(confidence)

        return return_per_t - alpha_quantile * volatility

    def value_at_risk_per_day(self):
        """
        Calculates the Value at Risk on a day by day basis.
        Using μ – z * σ See: https://wiki.hslu.ch/controlling/Value_at_Risk
        :return:
        """
        return self._value_at_risk(self.rate_of_return(), self.daily_volatility())

    def value_at_risk_total(self):
        return self._value_at_risk(self.holding_period_return(), self.total_volatility())
