import json
import os
from pathlib import Path

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from dqn.algorithm.Evaluation import Evaluation


class Plotter:

    @classmethod
    def plot(cls, agent_evaluation: Evaluation, result_path: Path):
        agent_evaluation.build_and_print_metrics()

        metrics = {
            "arithmetic_return": agent_evaluation.arithmetic_daily_return(),
            "logarithmic_return": agent_evaluation.logarithmic_daily_return(),
            "average_daily_return": agent_evaluation.average_daily_return(),
            "daily_return_variance": agent_evaluation.daily_return_variance(),
            "daily_return_variance_log": agent_evaluation.daily_return_variance("logarithmic"),
            "time_weighted_return": agent_evaluation.time_weighted_return(),
            "total_return": agent_evaluation.total_return(),
            "sharp_ratio": agent_evaluation.sharp_ratio(),
            "value_at_risk": agent_evaluation.value_at_risk(),
            "volatility": agent_evaluation.volatility(),
            "portfolio": agent_evaluation.get_daily_portfolio_value()[-1],
            "portfolio_market_delta": agent_evaluation.get_daily_portfolio_value()[-1] - agent_evaluation.data["close"].iloc[-1]
        }

        with open(os.path.join(result_path, 'metrics.json'), 'w') as f:
            f.write(json.dumps(metrics, indent=2))
           #f.write(str(metrics))

        portfolio_proxy = {'DQN-vanilla': {'portfolio_increase': agent_evaluation.get_daily_portfolio_value()}}
        plot_path = os.path.join(result_path, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        seaborn.set(rc={'figure.figsize': (15, 7)})
        seaborn.set_palette(seaborn.color_palette("Paired", 15))


        for model_name in portfolio_proxy.keys():
            first = True
            ax = None
            for key in portfolio_proxy[model_name]:
                profit_percentage = [
                    (portfolio_proxy[model_name][key][i] - portfolio_proxy[model_name][key][0]) /
                    portfolio_proxy[model_name][key][0] * 100
                    for i in range(len(portfolio_proxy[model_name][key]))]

                difference = len(portfolio_proxy[model_name][key]) - len(agent_evaluation.data)
                df = pd.DataFrame({'date': agent_evaluation.data.index,
                                   'portfolio': profit_percentage[difference:]})
                if not first:
                    df.plot(ax=ax, x='date', y='portfolio', label='Portfolio value')
                else:
                    ax = df.plot(x='date', y='portfolio', label='Portfolio value', color='darkblue')
                    first = False

            ax.set(xlabel='Time', ylabel='%Rate of Return')
            ax.set_title(f'Profit as Rate of Return on validation data')
            close_percentile = ((agent_evaluation.data['close'] / agent_evaluation.data['close'].iloc[0])*100)-100
            close_percentile.plot(ax=ax, x='date', y='close', label='Market value', color='orange')
            plt.legend()
            fig_file = os.path.join(plot_path, f'{model_name}.jpg')
            plt.savefig(fig_file, dpi=300)
