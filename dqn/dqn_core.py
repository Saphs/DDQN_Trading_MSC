import json
import logging
import os
from datetime import datetime
from pathlib import Path
import random

import torch
from pandas import DataFrame

from dqn.algorithm.chart_builder import ChartBuilder, Representation
from dqn.algorithm.config_parsing.dqn_config import DqnConfig
from dqn.algorithm.Evaluation import Evaluation
from dqn.algorithm.config_parsing.history import History, TrainingSession, custom_encode, custom_decode
from dqn.algorithm.data_loader import DataLoader
from dqn.algorithm.dqn_agent import DqnAgent
from dqn.algorithm.environment import Environment
from dqn.algorithm.model.neural_network import NeuralNetwork

_USE_CUDA = False
_DEVICE = torch.device("cuda" if _USE_CUDA else "cpu")


def set_seed(x: int):
    logging.info(f"Setting seed to: {x}")
    torch.manual_seed(x)
    random.seed(x)


class DqnGym:
    def __init__(self, out_path: Path, config: DqnConfig):
        self._runtime_start: datetime = datetime.now()
        self._config: DqnConfig = config
        self._result_path = Path(os.path.join(
            out_path,
            str(int(self._runtime_start.utcnow().timestamp()))
        ))
        self._cache_path = os.path.join(
            out_path,
            "cache"
        )

    def _load_env(self, stock_data: DataFrame) -> Environment:
        return Environment(
            stock_data,
            self._config.observation_space,
            'model_prediction',
            _DEVICE,
            n_step=self._config.agent.n_step,
            batch_size=self._config.batch_size,
            window_size=self._config.window_size,
            transaction_cost=self._config.environment.transaction_cost,
            initial_capital=self._config.environment.initial_capital
        )

    def _load_data(self, stock_name: str, lower: float = None, upper: float = None) -> DataFrame:
        logging.info(f"Loading data for stock: {stock_name}")
        data_loader = DataLoader(Path(self._cache_path), stock_name)
        if lower is not None:
            l_data, _ = data_loader.get_section(lower)
            logging.info(
                f"Finished loading stock data. (lower: {lower * 100}%) {stock_name=} From: {l_data.head(1).index.values[0]} To: {l_data.tail(1).index.values[0]}"
            )
            return l_data
        elif upper is not None:
            _, u_data = data_loader.get_section(1 - upper)
            logging.info(
                f"Finished loading stock data. (upper: {upper * 100}%) {stock_name=} From: {u_data.head(1).index.values[0]} To: {u_data.tail(1).index.values[0]}"
            )

            return u_data
        else:
            return data_loader.get()

    def _init_new_agent(self, state_size: int, name: str = None) -> DqnAgent:
        c = self._config.agent
        return DqnAgent(
            state_size,
            batch_size=self._config.batch_size,
            gamma=c.gamma,
            replay_memory_size=c.replay_memory_size,
            target_update=c.target_net_update_interval,
            n_step=c.n_step,
            name=name
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

        logging.info(f"Loaded existing agent model from: {agent_save_file}")

        return agent

    def _load_history(self, old_agent_path: Path, current_agent_name: str) -> History:
        """Load the history file of an old agent or return a freshly initialized one."""
        if old_agent_path is not None:
            history_file: Path = Path.joinpath(old_agent_path.parent, "history.json")
            if os.path.exists(history_file):
                with open(history_file) as f:
                    return json.loads(f.read(), object_hook=custom_decode)
        return History(current_agent_name, 0, [])


    def _save_context(self, t_start: datetime, t_end: datetime, history: History):
        history.append_session(TrainingSession(t_start, t_end, self._config))
        with open(os.path.join(self._result_path, 'history.json'), 'w') as fh:
            json_str: str = json.dumps(history, default=custom_encode, indent=2)
            fh.write(json_str)
        with open(os.path.join(self._result_path, 'config_used.json'), 'w') as fc:
            fc.write(str(self._config))

    def _interact(self, q_function: NeuralNetwork, env: Environment, action_column: str) -> DataFrame:
        with torch.no_grad():
            action_list = []
            env.__iter__()

            for i, batch in enumerate(env):
                try:
                    # net(batch) -> apply network to batch
                    # .max(1)[1] -> get index of max value
                    # .cpu().numpy() -> access on cpu if on cuda and cast to numpy array
                    action_list += list(q_function(batch).max(1)[1].cpu().numpy())
                except ValueError:
                    print("Error")
                    action_list += [1]

        def _to_action(x: int) -> str:
            return env.code_to_action[x]
        env.data[action_column] = list(map(_to_action, action_list))
        return env.data

    def train(self, stock_name: str, old_agent: Path = None) -> None:
        if not os.path.exists(self._result_path):
            os.makedirs(self._result_path)
        else:
            print("Result path already exists, stopping execution.")
            return

        # Load and prepare the data set
        training_data = self._load_data(stock_name, lower=0.8)

        # Initialize environments and agent using given data set
        t_env = self._load_env(training_data)
        if old_agent is None:
            logging.info("Initializing new DQN-Agent")
            agent = self._init_new_agent(t_env.state_size)
        else:
            logging.info(f"Initializing existing DQN-Agent: {old_agent.name}")
            agent = self._load_old_agent(t_env.state_size, old_agent)


        t0 = datetime.now()
        # Train agent (learn approximated *Q-Function)
        reward_df = agent.train(t_env, num_episodes=self._config.episodes)
        t1 = datetime.now()

        logging.info(f"Saving model under: {self._result_path}")
        agent.save_model(self._result_path)
        reward_df.to_csv(Path.joinpath(self._result_path, "rewards.csv"))
        history = self._load_history(old_agent, agent.name)
        self._save_context(t0, t1, history)

    def evaluate(self, stock_name: str, agent_file: Path) -> None:
        # Load and prepare the data set
        test_data: DataFrame = self._load_data(stock_name, upper=0.2)

        # Initialize environments and agent using given data set
        v_env: Environment = self._load_env(test_data)
        agent: DqnAgent = self._load_old_agent(v_env.state_size, agent_file)

        # Configure plotting and metric tool
        cb = ChartBuilder()
        cb.set_validation_path(agent_file.parent)

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

        # Build charts for training data (performance on known data)
        cb.clear_evaluations()
        known_data = self._load_data(stock_name, lower=0.8)
        t_env: Environment = self._load_env(known_data)
        ev_known_training_mode = Evaluation(agent, t_env, stock_name, self._config.window_size, "known", force_eval_mode=False)
        ev_known_evaluation_mode = Evaluation(agent, t_env, stock_name, self._config.window_size, "known", force_eval_mode=True)
        known_buy_and_hold = Evaluation(None, t_env, stock_name, self._config.window_size)

        cb.set_reference_evaluation(known_buy_and_hold)
        cb.add_evaluation(ev_known_training_mode)
        cb.add_evaluation(ev_known_evaluation_mode)

        cb.plot()
        cb.clear_evaluations()

        # Build data set overview chart
        cb.plot_data(known_data, test_data, stock_name)


