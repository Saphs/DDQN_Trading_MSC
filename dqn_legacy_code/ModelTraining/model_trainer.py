import json
import os
from pathlib import Path

from datetime import datetime
import pandas as pd
import torch

from dqn_legacy_code.DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from dqn_legacy_code.DataLoader.DataLoader import YahooFinanceDataLoader
from dqn_legacy_code.ConfigClasses.loader_definitions import DATA_LOADERS
from dqn_legacy_code.ConfigClasses.model_config import ModelConfig
from dqn_legacy_code.DeepQNetwork.Agent import Agent
from dqn_legacy_code.DeepQNetwork.DqnAgent import DqnAgent
from dqn_legacy_code.ModelTraining.plotter import Plotter


class ModelTrainer:
    active_data_set: str = ""
    active_data_loader: YahooFinanceDataLoader = None
    active_training_data: pd.DataFrame = None

    def __init__(self, config: ModelConfig):
        self._runtime_start: datetime = datetime.now()
        self._device = torch.device("cuda" if config.cuda else "cpu")
        self._config: ModelConfig = config
        self._result_path = os.path.join(
            self._config.training_params.result_base_path,
            str(int(self._runtime_start.utcnow().timestamp()))
        )

    def load_data_set(self):
        self.active_data_set = self._config.training_params.data_set_name
        print(f"[{self.active_data_set=}] Loading data set...", end='')
        self.active_data_loader = DATA_LOADERS[self.active_data_set]
        self.active_training_data = self.active_data_loader.data_train
        print(" Done!")
        print(f"Training data sample:\n{self.active_training_data}")

    def _load_envs(self):
        training_env: DataAutoPatternExtractionAgent = DataAutoPatternExtractionAgent(
            self.active_training_data,
            self._config.state_mode,
            'action_auto_pattern_extraction',
            self._device,
            n_step=self._config.training_params.n_step,
            batch_size=self._config.training_params.batch_size,
            window_size=self._config.training_params.window_size,
            transaction_cost=self._config.training_params.transaction_cost
        )
        return training_env

    def _init_agent(self, state_size):
        return DqnAgent(
            state_size,
            self.active_data_set,
            self._config.state_mode,
            self._config.training_params.transaction_cost,
            BATCH_SIZE=self._config.training_params.batch_size,
            GAMMA=self._config.training_params.gamma,
            ReplayMemorySize=self._config.training_params.replay_memory_size,
            TARGET_UPDATE=self._config.training_params.target_net_update_interval,
            n_step=self._config.training_params.n_step
        )

    def save_context(self):
        with open(os.path.join(self._result_path, 'config_used.json'), 'w') as f:
            f.write(str(self._config))

    def start(self) -> None:
        if not os.path.exists(self._result_path):
            os.makedirs(self._result_path)
        else:
            print("Result path already exists, stopping execution.")
            return

        # Load the selected data set
        self.load_data_set()

        # Load environments and initialize agent
        t_env = self._load_envs()
        agent = self._init_agent(t_env.state_size)

        # Train model (learn approximated *Q-Function)
        agent.train(t_env, num_episodes=self._config.training_params.episodes)
        agent.save_model(self._result_path)
        self.save_context()








