import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import seaborn
from matplotlib import pyplot as plt, lines
from pandas import DataFrame, Series
from sympy.physics.control.control_plots import matplotlib

from dqn.algorithm.Evaluation import Evaluation


class ChartBuilder:

    def __init__(self):
        self._reference_evaluation: Optional[Evaluation] = None
        self._evaluations: List[Evaluation] = []
        self._validation_path: Optional[Path] = None

    def add_evaluation(self, ev: Evaluation):
        self._evaluations.append(ev)

    def set_reference_evaluation(self, ev: Evaluation):
        self._reference_evaluation = ev
        #  This kind of overreaches the logical bounds as it technically changes the evaluation.
        old_name = self._reference_evaluation.metric_table['model_name'].iloc[0]
        self._reference_evaluation.metric_table['model_name'].iloc[0] = f"(*) {old_name}"

    def set_validation_path(self, result_path: Path):
        self._validation_path = os.path.join(result_path, 'validation')
        if not os.path.exists(self._validation_path):
            os.makedirs(self._validation_path)

    def save_metrics(self) -> DataFrame:
        evals = self._evaluations
        evals.insert(0, self._reference_evaluation)
        frames = list(map(lambda ev: ev.calculate_metric_table(), evals))
        df = pd.concat(frames).reset_index(drop=True)
        metrics_file = os.path.join(self._validation_path, f'metrics.csv')
        df.to_csv(metrics_file)
        logging.info(f"Saved metrics file {metrics_file}")
        return df

    def plot(self):
        for ev in self._evaluations:
            self._plot(ev)

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
        ax.set_title(f'Agent-{ev.model_name} (in {mode}-mode) compared to market ({self._reference_evaluation.model_name})')

        plt.legend()
        fig_file = os.path.join(self._validation_path, f'perf_{ev.model_name}_{mode}.jpg')
        plt.savefig(fig_file, dpi=300)
        logging.info(f"Saved figure {fig_file}")

