import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import random

import pandas as pd
import torch
from pandas import DataFrame
from torch._C._profiler import ProfilerActivity
from torch.profiler import profile

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.ddqn_agent import DDqnAgent
from agent.chart_builder import ChartBuilder
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.Evaluation import Evaluation
from agent.algorithm.config_parsing.history import History, TrainingSession, custom_encode, custom_decode
from data_access.data_loader import DataLoader
from agent.algorithm.dqn_agent import DqnAgent
from agent.algorithm.environment import Environment
from agent.algorithm.model.neural_network import NeuralNetwork

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logging.info(f"Cuda is available, setting device to: \033[96m{_DEVICE=}\033[0m")
else:
    logging.warning(f"Cuda is NOT available, using CPU at: {_DEVICE=}")

_STYLE_DQN = "dqn"
_STYLE_DDQN = "ddqn"

_HISTORY_FILE = "history.json"
_CONFIG_USED_FILE = "config_used.json"
_REWARDS_FILE = "rewards.csv"


def set_seed(x: int):
    logging.info(f"Setting seed to: {x}")
    torch.manual_seed(x)
    random.seed(x)


class DqnGym:
    def __init__(self, out_path: Path, config: DqnConfig):
        self._runtime_start: datetime = datetime.now()
        self._config: DqnConfig = config
        self.result_path = Path(os.path.join(
            out_path,
            str(int(self._runtime_start.utcnow().timestamp()))
        ))
        self._cache_path = os.path.join(
            out_path,
            "cache"
        )

    def _load_env(self, stock_data: DataFrame) -> Environment:
        return Environment(stock_data, self._config, device=_DEVICE)

    def _load_data(self, stock_name: str, split: float = None) -> tuple[DataFrame, DataFrame]:
        logging.info(f"Loading data for stock: {stock_name}")
        data_loader = DataLoader(Path(self._cache_path), stock_name)
        if split is not None:
            l_data, u_data = data_loader.get_section(split)
            logging.info(
                f"Finished loading stock data. (split {split * 100}% / {(1 - split) * 100}%) {stock_name=} start: {l_data.head(1).index.values[0]} split: {l_data.tail(1).index.values[0]} end: {u_data.tail(1).index.values[0]}"
            )
            return l_data, u_data
        else:
            return data_loader.get(), DataFrame()

    def _init_new_agent(self, state_size: int, name: str = None) -> DqnAgent:
        if self._config.agent.style == _STYLE_DQN:
            return DqnAgent.init(state_size, self._config, name=name)
        elif self._config.agent.style == _STYLE_DDQN:
            return DDqnAgent.init(state_size, self._config, name=name)
        else:
            raise ValueError(
                f"Failed to load agent for unknown style: {self._config.agent.style}. Possible style are: [\"{_STYLE_DQN}\", \"{_STYLE_DDQN}\"]"
            )

    def _load_old_agent(self, state_size: int, agent_save_file: Path) -> DqnAgent:
        name_id = agent_save_file.name.lstrip('model_').rstrip('.pkl')
        agent = self._init_new_agent(state_size, name=name_id)

        saved_policy_net = NeuralNetwork(state_size)
        saved_policy_net.load_state_dict(torch.load(agent_save_file))
        saved_policy_net.to(_DEVICE)

        agent.policy_net = saved_policy_net
        agent.policy_net.train()
        agent.target_net = saved_policy_net
        agent.target_net.to(_DEVICE)

        logging.info(f"Loaded existing agent model from: {agent_save_file}")

        return agent

    def _load_history(self, old_agent_path: Path, current_agent_name: str) -> History:
        """Load the history file of an old agent or return a freshly initialized one."""
        if old_agent_path is not None:
            history_file: Path = Path.joinpath(old_agent_path.parent, _HISTORY_FILE)
            if os.path.exists(history_file):
                with open(history_file) as f:
                    return json.loads(f.read(), object_hook=custom_decode)
        return History(current_agent_name, 0, [])

    def _save_context(self, t_start: datetime, t_end: datetime, history: History, steps_done: int):
        history.append_session(TrainingSession(t_start, t_end, self._config))
        with open(os.path.join(self.result_path, _HISTORY_FILE), 'w') as fh:
            json_str: str = json.dumps(history, default=custom_encode, indent=2)
            fh.write(json_str)
        with open(os.path.join(self.result_path, _CONFIG_USED_FILE), 'w') as fc:
            c = copy.deepcopy(self._config)
            fc.write(str(c))
        cb = ChartBuilder()
        cb.set_target_path(self.result_path)
        cb.plot_epsilon(self._config.agent, steps_done)

    def train(self, stock_name: str, old_agent: Path = None, name: str = None) -> None:
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        else:
            print("Result path already exists, stopping execution.")
            return

        # Load and prepare the data set
        training_data, _ = self._load_data(stock_name, split=0.8)

        # Initialize environments and agent using given data set
        t_env = self._load_env(training_data)
        if old_agent is None:
            logging.info(f"Initializing new {self._config.agent.style}-Agent")
            agent = self._init_new_agent(t_env.state_size, name)
        else:
            logging.info(f"Initializing existing {self._config.agent.style}-Agent: {old_agent.name}")
            agent = self._load_old_agent(t_env.state_size, old_agent)
        agent.set_target_path(self.result_path)

        t0 = datetime.now()
        # Train agent (learn approximated *Q-Function)
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        reward_df = agent.train(t_env, num_episodes=self._config.episodes)
        #path = os.path.join(self.result_path, f'final/profiling.text')
        #with open(path, 'w') as f:
        #    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=500))
        t1 = datetime.now()

        # Save auxiliary data
        reward_df.to_csv(Path.joinpath(self.result_path, _REWARDS_FILE), index=False)
        history = self._load_history(old_agent, agent.name)

        self._save_context(t0, t1, history, agent.steps_done)

    def evaluate(self, stock_name: str, agent_file: Path) -> None:
        # override any transaction cost to set a reasonable evaluation cost
        old_costs = self._config.environment.transaction_cost
        self._config.environment.transaction_cost = 0.001

        # Load and prepare the data set
        training_data, validation_data = self._load_data(stock_name, split=0.8)

        # Initialize environments and agent using given data set
        v_env: Environment = self._load_env(validation_data)
        agent: DqnAgent = self._load_old_agent(v_env.state_size, agent_file)

        # Configure plotting and metric tool
        cb = ChartBuilder()
        validation_path = Path(os.path.join(agent_file.parent, 'validation'))
        cb.set_target_path(validation_path)

        # Run evaluations & add to metric tool
        ev_training_mode = Evaluation(agent, v_env, stock_name, self._config.window_size, "valid", force_eval_mode=False)
        ev_evaluation_mode = Evaluation(agent, v_env, stock_name, self._config.window_size, "valid", force_eval_mode=True)
        buy_and_hold = Evaluation(None, v_env, stock_name, self._config.window_size)

        cb.set_reference_evaluation(buy_and_hold)
        cb.add_evaluation(ev_training_mode)
        cb.add_evaluation(ev_evaluation_mode)

        # Save results
        cb.plot()
        print(cb.save_metrics())
#        print(ev_evaluation_mode.back_test.run())

        # Build charts for training data (performance on known data)
        cb.clear_evaluations()
        t_env: Environment = self._load_env(training_data)
        ev_known_training_mode = Evaluation(agent, t_env, stock_name, self._config.window_size, "known", force_eval_mode=False)
        ev_known_evaluation_mode = Evaluation(agent, t_env, stock_name, self._config.window_size, "known", force_eval_mode=True)
        known_buy_and_hold = Evaluation(None, t_env, stock_name, self._config.window_size)

        self._config.environment.transaction_cost = old_costs
        ref_env = self._load_env(training_data)
        ref = Evaluation(None, ref_env, stock_name, self._config.window_size)
        ref.raw_data.tail(1).to_csv(validation_path.joinpath("test_reference_raw.csv"))
        old_costs = self._config.environment.transaction_cost
        self._config.environment.transaction_cost = 0.001

        cb.set_reference_evaluation(known_buy_and_hold)
        cb.add_evaluation(ev_known_training_mode)
        cb.add_evaluation(ev_known_evaluation_mode)

        cb.plot()
        cb.clear_evaluations()

        # Build data set overview chart
        cb.plot_data(training_data, validation_data, stock_name)

        # Load and plot rewards gathered
        reward_df = pd.read_csv(agent_file.parent.parent.joinpath(_REWARDS_FILE))
        cb.plot_rewards(reward_df, agent.target_update)
        cb.plot_loss(reward_df, agent.target_update)
        cb.plot_capital(reward_df, agent.target_update)

        # Reset overridden value for transaction cost
        self._config.environment.transaction_cost = old_costs

    def check_point_eval(self, stock_name: str, checkpoints: list[Path]):
        if len(checkpoints) < 1:
            raise ValueError("List of checkpoints was empty.")

        out_folder = checkpoints[0].parent.joinpath("eval")
        eval_mode_folder = out_folder.joinpath("eval_mode")
        train_mode_folder = out_folder.joinpath("train_mode_folder")
        os.makedirs(eval_mode_folder, exist_ok=True)
        os.makedirs(train_mode_folder, exist_ok=True)

        ws = self._config.window_size
        training_data, validation_data = self._load_data(stock_name, split=0.8)
        t_env: Environment = self._load_env(training_data)
        buy_and_hold = Evaluation(None, t_env, stock_name, ws)

        cb = ChartBuilder()
        cb.set_reference_evaluation(buy_and_hold)
        for a_path in checkpoints:
            a = self._load_old_agent(t_env.state_size, a_path)

            tr_mode = Evaluation(a, t_env, stock_name, ws, force_eval_mode=False)
            plt = cb.gen_plot(tr_mode)
            plt.savefig(train_mode_folder.joinpath(f"{a.name}.jpg"), dpi=300)
            plt.close()

            ev_mode = Evaluation(a, t_env, stock_name, ws, force_eval_mode=True)
            plt = cb.gen_plot(ev_mode)
            plt.savefig(eval_mode_folder.joinpath(f"{a.name}.jpg"), dpi=300)
            plt.close()






