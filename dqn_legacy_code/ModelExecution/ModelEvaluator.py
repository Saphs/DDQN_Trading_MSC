import os

import pandas as pd
import torch

from dqn_legacy_code.DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from dqn_legacy_code.DataLoader.DataLoader import YahooFinanceDataLoader
from dqn_legacy_code.VanillaInput.DeepQNetwork import NeuralNetwork
from dqn_legacy_code.PatternDetectionInCandleStick.Evaluation import Evaluation
from dqn_legacy_code.ConfigClasses.loader_definitions import DATA_LOADERS
from dqn_legacy_code.ConfigClasses.model_config import ModelConfig
from dqn_legacy_code.ModelTraining.plotter import Plotter


class ModelEvaluator:
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

    def load_data_set(self):
        self.active_data_set = self._config.evaluation_params.data_set_name
        print(f"[{self.active_data_set=}] Loading data set...", end='')
        self.active_data_loader = DATA_LOADERS[self.active_data_set]
        self.active_validation_data = self.active_data_loader.data_test
        self.active_validation_data.head(50).to_csv(os.path.join(self.model_dir, 'eval_data_sample.csv'))
        print(" Done!")
        #print(f"Validation data sample:\n{self.active_validation_data}")

    def _load_envs(self):
        validation_env: DataAutoPatternExtractionAgent = DataAutoPatternExtractionAgent(
            self.active_validation_data,
            self._config.state_mode,
            'action_auto_pattern_extraction',
            self._device,
            n_step=self._config.evaluation_params.n_step,
            batch_size=self._config.evaluation_params.batch_size,
            window_size=self._config.evaluation_params.window_size,
            transaction_cost=self._config.evaluation_params.transaction_cost
        )

        pd.DataFrame(validation_env.states).head(50).to_csv(
            os.path.join(self.model_dir, 'eval_state_sample.csv')
        )
        return validation_env

    def _evaluate(self, network: NeuralNetwork, environment: DataAutoPatternExtractionAgent, initial_investment=1000) -> Evaluation:
        action_list = []
        environment.__iter__()
        #odict_keys(['policy_network.0.weight', 'policy_network.0.bias', 'policy_network.1.weight', 'policy_network.1.bias', 'policy_network.1.running_mean', 'policy_network.1.running_var', 'policy_network.1.num_batches_tracked', 'policy_network.2.weight', 'policy_network.2.bias', 'policy_network.3.weight', 'policy_network.3.bias', 'policy_network.3.running_mean', 'policy_network.3.running_var', 'policy_network.3.num_batches_tracked', 'policy_network.4.weight', 'policy_network.4.bias'])

       # print(network.state_dict()['policy_network.1.running_var'])

        for i, batch in enumerate(environment):
            if i % 5 == 0:
                pass
                #print(f"{i=}{batch}")

            try:
                action_batch = network(batch).max(1)[1]
                action_list += list(action_batch.cpu().numpy())
            except ValueError:
                action_list += [1]

        print(action_list)
        environment.make_investment(action_list)
        ev = Evaluation(
            environment.data,
            environment.action_name,
            initial_investment,
            self._config.evaluation_params.transaction_cost
        )
        return ev

    def load_model(self, state_size) -> NeuralNetwork:
        net = NeuralNetwork(state_size, 3)
        net.load_state_dict(torch.load(self.model_path))
        net.to(self._device)
        #net.eval()
        return net

    def evaluate(self):
        # Load the selected data set
        self.load_data_set()

        print(f"Validation data sample:\n{self.active_validation_data}")
        # Load environments and initialize agent
        v_env = self._load_envs()
        network = self.load_model(v_env.state_size)
        evaluation = self._evaluate(network, v_env)
        Plotter.plot(evaluation, self.active_data_loader, self.model_dir)

