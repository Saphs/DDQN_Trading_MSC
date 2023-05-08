import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from pandas import DataFrame

from data_access.stock_dao import StockDao
from dqn.algorithm.config_parsing.dqn_config import DqnConfig
from dqn.algorithm.Evaluation import Evaluation
from dqn.algorithm.data_loader import DataLoader
from dqn.algorithm.dqn_agent import DqnAgent
from dqn.algorithm.environment import Environment
from dqn.algorithm.model.neural_network import NeuralNetwork
from dqn.algorithm.plotter import Plotter

_USE_CUDA = False
_DEVICE = torch.device("cuda" if _USE_CUDA else "cpu")

class DqnGym:
    def __init__(self, out_path: Path, config: DqnConfig):
        self._runtime_start: datetime = datetime.now()
        self._config: DqnConfig = config
        self._stock_dao = StockDao()
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
            'action_auto_pattern_extraction',
            _DEVICE,
            n_step=self._config.agent.n_step,
            batch_size=self._config.batch_size,
            window_size=self._config.window_size,
            transaction_cost=self._config.environment.transaction_cost
        )

    def _load_data(self, stock_name: str, lower:float = None, upper: float = None) -> DataFrame:
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
                f"Finished loading stock data. (upper: {(1-upper) * 100}%) {stock_name=} From: {u_data.head(1).index.values[0]} To: {u_data.tail(1).index.values[0]}"
            )
            return u_data
        else:
            raise ValueError(f"Failed to load stock: {stock_name}. {lower=} and {upper=} portions were inconclusive.")



    def _init_new_agent(self, state_size: int) -> DqnAgent:
        c = self._config.agent
        return DqnAgent(
            state_size,
            batch_size=self._config.batch_size,
            gamma=c.gamma,
            replay_memory_size=c.replay_memory_size,
            target_update=c.target_net_update_interval,
            n_step=c.n_step
        )

    def _load_old_agent(self, state_size: int, agent_save_file: Path, evaluation_mode: bool = False) -> DqnAgent:
        agent = self._init_new_agent(state_size)

        saved_policy_net = NeuralNetwork(state_size)
        saved_policy_net.load_state_dict(torch.load(agent_save_file))
        saved_policy_net.to(_DEVICE)

        agent.policy_net = saved_policy_net
        agent.policy_net.train()
        agent.target_net = saved_policy_net

        logging.debug(f"Loaded existing agent model from: {agent_save_file}")

        if evaluation_mode:
            pass
            # Freezes the network weights
            #agent.policy_net.eval()
            #agent.target_net.eval()

        return agent

    def _save_context(self):
        with open(os.path.join(self._result_path, 'config_used.json'), 'w') as f:
            f.write(str(self._config))

    def _evaluate(self, agent: DqnAgent, environment: Environment, initial_investment=1000) -> Evaluation:
        # Take the agents policy network (Q*-Function)
        network: NeuralNetwork = NeuralNetwork(environment.state_size).to(environment.device)
        network.load_state_dict(agent.policy_net.state_dict())
        with torch.no_grad():
            #network.eval()
            action_list = []
            environment.__iter__()

            first = True
            for batch in environment:
                if first:
                    print(batch)
                    first = False
                try:
                    action_batch = network(batch).max(1)[1]
                    action_list += list(action_batch.cpu().numpy())
                except ValueError:
                    print("Error")
                    action_list += [1]

            environment.make_investment(action_list)
            ev = Evaluation(
                environment.data,
                environment.action_name,
                initial_investment,
                self._config.environment.transaction_cost
            )
            return ev

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
            agent = self._init_new_agent(t_env.state_size)
        else:
            agent = self._load_old_agent(t_env.state_size, old_agent)

        # Train agent (learn approximated *Q-Function)
        agent.train(t_env, num_episodes=self._config.episodes)
        logging.info(f"Saving model under: {self._result_path}")
        agent.save_model(self._result_path)
        self._save_context()

    def evaluate(self, stock_name: str, agent_file: Path) -> None:
        # Load and prepare the data set
        test_data = self._load_data(stock_name, upper=0.2)

        # Initialize environments and agent using given data set
        v_env: Environment = self._load_env(test_data)
        agent: DqnAgent = self._load_old_agent(v_env.state_size, agent_file, evaluation_mode=True)
        evaluation = self._evaluate(agent, v_env)
        Plotter.plot(evaluation, agent_file.parent)
