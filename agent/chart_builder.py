import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from enum import Enum
from sympy.physics.control.control_plots import matplotlib

from agent.Evaluation import Evaluation
from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig


class Representation(Enum):
    EIGHTY_TWENTY_SPLIT = 1


class ChartBuilder:

    def __init__(self):
        self._reference_evaluation: Optional[Evaluation] = None
        self._evaluations: List[Evaluation] = []
        self._target_path: Optional[Path] = None

    def add_evaluation(self, ev: Evaluation):
        self._evaluations.append(ev)

    def set_reference_evaluation(self, ev: Evaluation):
        self._reference_evaluation = ev
        #  This kind of overreaches the logical bounds as it technically changes the evaluation.
        old_name = self._reference_evaluation.metric_table['model_name'].iloc[0]
        self._reference_evaluation.metric_table['model_name'].iloc[0] = f"(*) {old_name}"

    def set_target_path(self, result_path: Path):
        self._target_path = result_path
        if not os.path.exists(self._target_path):
            os.makedirs(self._target_path)

    def clear_evaluations(self):
        self._reference_evaluation: Optional[Evaluation] = None
        self._evaluations: List[Evaluation] = []

    def save_metrics(self) -> DataFrame:
        evals = self._evaluations
        evals.insert(0, self._reference_evaluation)
        frames = list(map(lambda ev: ev.calculate_metric_table(), evals))
        df = pd.concat(frames).reset_index(drop=True)
        metrics_file = os.path.join(self._target_path, f'metrics.csv')
        df.to_csv(metrics_file)
        logging.info(f"Saved metrics file {metrics_file}")
        return df

    def plot(self):
        for ev in self._evaluations:
            self._plot(ev)

    def plot_data(self, data_low: DataFrame, data_high: DataFrame, stock_name: str):
        # Set some configuration values
        seaborn.set(rc={'figure.figsize': (15, 7)})
        seaborn.set_palette(seaborn.color_palette("tab10"))

        tmp_df = pd.concat([data_low, data_high])
        tmp_df['training_data'] = data_low['close']
        tmp_df['validation_data'] = data_high['close']
        tmp_df = tmp_df.reset_index()[['date', 'training_data', 'validation_data']]

        ax: matplotlib.axes.Axes = tmp_df.plot(x='date')
        ax.set(xlabel='Time', ylabel='Close value')
        ax.set_title(f'Close value {stock_name}-stock over time, training and validation split.')
        plt.legend()
        fig_file = os.path.join(self._target_path, f'raw_stock.jpg')
        plt.savefig(fig_file, dpi=300)

    def plot_rewards(self, reward_df: DataFrame):
        reward_df['sma_reward'] = reward_df['avg_reward'].rolling(window=10).mean()
        r_df = reward_df[['episode', 'sma_reward', 'avg_reward']]
        ax: matplotlib.axes.Axes = r_df.plot(x='episode')
        ax.set(xlabel='Episode', ylabel='Average reward')
        ax.set_title(f'Avg rewards of model per episode trained')
        plt.legend()
        fig_file = os.path.join(self._target_path, f'rewards.jpg')
        plt.savefig(fig_file, dpi=300)

    def plot_loss(self, loss_df: DataFrame):
        ax: matplotlib.axes.Axes = loss_df.plot(x='episode', y='avg_loss')
        ax.set(xlabel='Episode', ylabel='Average loss')
        ax.set_title(f'Avg loss of model per episode trained')
        plt.legend()
        fig_file = os.path.join(self._target_path, f'loss.jpg')
        plt.savefig(fig_file, dpi=300)

    def plot_epsilon(self, ap: AgentParameters, steps_done: int):
        self._plot_epsilon(
            # Apparently python does not allow me to use .agent as a object. Fix this.
            ap.epsilon_start,
            ap.epsilon_end,
            ap.epsilon_decay,
            steps_done
        )

    def _plot_epsilon(self, e_start: float, e_end: float, e_decay: float, steps_done: int):
        # Calculate epsilon values
        steps = np.arange(steps_done)
        epsilons = [e_end + (e_start - e_end) * np.exp(-1. * step / e_decay) for step in steps]

        # Plot epsilon values
        plt.plot(steps, epsilons)
        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        plt.title('Decay of Epsilon')
        plt.grid(True)
        fig_file = os.path.join(self._target_path, f'epsilon_decay.jpg')
        plt.savefig(fig_file, dpi=300)
        logging.info(f"Saved epsilon decay under {fig_file}")

    def _as_percent(self, s: Series, start_value: Optional[float] = None) -> Series:
        if start_value is not None:
            return ((s / start_value) * 100) - 100
        else:
            return ((s / s[0]) * 100) - 100

    def _plot(self, ev: Evaluation):
        # Set some configuration values
        seaborn.set(rc={'figure.figsize': (15, 7)})
        seaborn.set_palette(seaborn.color_palette("tab10"))

        # Line chart
        target_data = ev.raw_data['portfolio']
        target_perc = self._as_percent(target_data)

        ref_data = self._reference_evaluation.raw_data['portfolio']
        ref_perc = self._as_percent(ref_data)

        df = pd.DataFrame({
            'date': ev.raw_data['date'],
            ev.metric_table['model_name'][0]: target_perc,
            'Buy & Hold': ref_perc,
        })
        ax: matplotlib.axes.Axes = df.plot(x='date')

        # Transaction markers
        eff_transactions = ev.raw_data[ev.raw_data['effective_transaction']]
        buy_transactions = eff_transactions[eff_transactions['prediction'].str.contains("buy")]
        y0 = self._reference_evaluation.raw_data['portfolio'].iloc[0]
        buy_transactions_perc = self._as_percent(buy_transactions['portfolio'], y0)
        ax.vlines(
            x=buy_transactions.index,
            ymin=buy_transactions_perc,
            ymax=buy_transactions_perc + 3,
            colors=["green"],
            linewidth=0.5
        )
        ax.scatter(
            x=buy_transactions.index,
            y=buy_transactions_perc + 3,
            marker="v",
            color="green"
        )

        sell_transactions = eff_transactions[eff_transactions['prediction'].str.contains("sell")]
        sell_transactions_perc = self._as_percent(sell_transactions['portfolio'], y0)
        ax.vlines(
            x=sell_transactions.index,
            ymin=sell_transactions_perc,
            ymax=sell_transactions_perc + 3,
            colors=["red"],
            linewidth=0.5
        )
        ax.scatter(
            x=sell_transactions.index,
            y=sell_transactions_perc + 3,
            marker="^",
            color="red"
        )

        # Meta values
        ax.set(xlabel='Time', ylabel='Return in %')
        mode = "Eval" if ev.evaluation_mode else "Train"
        ax.set_title(f'Agent-{ev.model_name} (in {mode}-mode) compared to market ({self._reference_evaluation.model_name}), Stock: {self._reference_evaluation.stock_name}')

        plt.legend()
        fig_file = os.path.join(self._target_path, f'{ev.name_prefix}_{ev.model_name}_{mode}.jpg')
        plt.savefig(fig_file, dpi=300)
        logging.info(f"Saved figure {fig_file}")

