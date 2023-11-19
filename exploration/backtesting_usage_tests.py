from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG
from pandas import DataFrame

from agent import gym_core
from agent.Evaluation import Evaluation
from agent.algorithm.config_parsing.dqn_config_codec import DqnConfigCodec
from agent.algorithm.ddqn_agent import DDqnAgent
from agent.algorithm.dqn_agent import DqnAgent
from agent.algorithm.environment import Environment
from agent.algorithm.model.neural_network import NeuralNetwork
from data_access.data_loader import DataLoader

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_STYLE_DQN = "dqn"
_STYLE_DDQN = "ddqn"

pd.options.mode.chained_assignment = None

def _init_new_agent(state_size: int, config, name: str = None) -> DqnAgent:
    if config.agent.style == _STYLE_DQN:
        return DqnAgent.init(state_size, config, name=name)
    elif config.agent.style == _STYLE_DDQN:
        return DDqnAgent.init(state_size, config, name=name)
    else:
        raise ValueError(
            f"Failed to load agent for unknown style: {config.agent.style}. Possible style are: [\"{_STYLE_DQN}\", \"{_STYLE_DDQN}\"]"
        )


def _load_old_agent(state_size: int, config, agent_save_file: Path) -> DqnAgent:
    name_id = agent_save_file.name.lstrip('model_').rstrip('.pkl')
    agent = _init_new_agent(state_size, config, name=name_id)

    saved_policy_net = NeuralNetwork(state_size)
    saved_policy_net.load_state_dict(torch.load(agent_save_file))
    saved_policy_net.to(_DEVICE)

    agent.policy_net = saved_policy_net
    agent.policy_net.train()
    agent.target_net = saved_policy_net
    agent.target_net.to(_DEVICE)

    print(f"Loaded existing agent model from: {agent_save_file}")

    return agent


def _load_data(stock_name: str, split: float = None) -> tuple[DataFrame, DataFrame]:
    print(f"Loading data for stock: {stock_name}")
    data_loader = DataLoader(Path("""C:\\Users\\tizia\PycharmProjects\\DDQN_Trading_MSC\\out\\baseline"""), stock_name,
                             skip_cache=True, db_config="../db_config.json")
    if split is not None:
        l_data, u_data = data_loader.get_section(split)
        print(
            f"Finished loading stock data. (split {split * 100}% / {(1 - split) * 100}%) {stock_name=} start: {l_data.head(1).index.values[0]} split: {l_data.tail(1).index.values[0]} end: {u_data.tail(1).index.values[0]}"
        )
        return l_data, u_data
    else:
        return data_loader.get(), DataFrame()


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


def run():
    path = Path("C:\\Users\\tizia\\PycharmProjects\\DDQN_Trading_MSC\\agent\\hyper_parameters.json")
    agent_file = Path(
        "C:\\Users\\tizia\\PycharmProjects\\DDQN_Trading_MSC\\out\\DDQN_buf256_sam30\\1695558457\\final\\best_455_for-seed_1234567892.pkl")
    config = DqnConfigCodec.read_json(path)
    print(config)
    _, test_data = _load_data(stock_name="Merck", split=0.8)
    test_data_for_bt = test_data.rename(columns={"open": "Open", "close": "Close", "high": "High", "low": "Low"})

    test_data_for_bt = test_data_for_bt[["Open", "High", "Low", "Close"]]
    test_data_for_bt.index.name = None

    # Initialize environments and agent using given data set
    v_env: Environment = Environment(test_data, config, device=_DEVICE)
    agent: DqnAgent = _load_old_agent(v_env.state_size, config, agent_file)
    v_eval: Evaluation = Evaluation(agent, v_env, config.environment.data_set_name, config.window_size, "valid", force_eval_mode=False)

    print(v_eval._predict())

    class Predefined(Strategy):

        actions = v_eval._predict()['prediction_numeric']

        def init(self):
            pass
        def next(self):
            #print(self.data.index[-1], type(self.data.index[-1]))
            try:
                p_action = (self.actions[self.data.index[-1]])
                if p_action == 0:
                    self.buy()
                elif p_action == 1:
                    pass
                else:
                    self.sell()
            except KeyError as err:
                print(f"KeyError: {err}")

    bt = Backtest(test_data_for_bt, Predefined, cash=10000, commission=0.001, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot()


if __name__ == "__main__":
    run()
