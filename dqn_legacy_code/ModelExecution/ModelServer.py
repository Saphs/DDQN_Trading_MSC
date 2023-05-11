import http.server
import os
import typing

import pandas as pd
import torch
from torch import tensor

from dqn_legacy_code.DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from dqn_legacy_code.DataLoader.DataLoader import YahooFinanceDataLoader
from dqn_legacy_code.VanillaInput.DeepQNetwork import NeuralNetwork
from dqn_legacy_code.ConfigClasses.model_config import ModelConfig

PORT = 8888
Handler = http.server.SimpleHTTPRequestHandler


class ModelServer:
    active_data_set: str = ""
    active_data_loader: YahooFinanceDataLoader = None
    active_validation_data: pd.DataFrame = None

    def __init__(self, config: ModelConfig):
        self._device = torch.device("cuda" if config.cuda else "cpu")
        self._config: ModelConfig = config
        self.model_path: str = self._get_model_path()

    def _get_model_path(self) -> str:
        base = self._config.evaluation_params.model_base_path
        model_folder = self._config.evaluation_params.model_folder_name
        if model_folder == "@latest":
            numeric_folder_names = list(filter(lambda s: s.isnumeric(), os.listdir(base)))
            numeric_folder_names.sort()
            model_folder = str(numeric_folder_names[-1])
            print(f"Latest model found in folder: {model_folder=}")
        self.model_dir = os.path.join(base, model_folder)
        model_name = list(filter(lambda s: s.endswith('.pkl'), os.listdir(self.model_dir)))[0]
        return os.path.join(self.model_dir, model_name)

    def load_model(self, state_size) -> NeuralNetwork:
        net = NeuralNetwork(state_size, 3)
        net.load_state_dict(torch.load(self.model_path))
        net.to(self._device)
        return net

    def _load_pseudo_env(self):
        pseudo_env: DataAutoPatternExtractionAgent = DataAutoPatternExtractionAgent(
            self.active_validation_data,
            self._config.state_mode,
            'action_auto_pattern_extraction',
            self._device,
            n_step=self._config.evaluation_params.n_step,
            batch_size=1,
            window_size=self._config.evaluation_params.window_size,
            transaction_cost=self._config.evaluation_params.transaction_cost
        )
        return pseudo_env

    def _act(self, state: typing.List[float]) -> (int, str):
        action_to_action_name = ["buy", 'none', 'sell']
        t = tensor([state])
        action_id = self.model(t).max(1)[1].item()
        return action_id, action_to_action_name[action_id]


    def serve(self):
        # Hacky mapping from state_mode to state_size. This should be done properly
        mode_to_size = [0, 4, 5, 8, 3]
        self.model = self.load_model(mode_to_size[self._config.state_mode])
        self.model.eval()

        print(self._act([0.6071, 0.6160, 0.6156, 0.6255, 1.0000, 0.8794, 0.1206, 0.0000]))

