import json

import torch

from dqn_legacy_code.ConfigClasses.evaluation_params import EvaluationParams
from dqn_legacy_code.ConfigClasses.training_params import TrainingParams


class ModelConfig:

    # Whether CUDA should be used (Nvidia hardware ML-Acceleration)
    cuda: bool = False
    # Defines the mode of operation for the model. 0 = Training, 1 = Execution
    execution_type: int = 0
    state_mode: int = 1
    training_params: TrainingParams = {}
    evaluation_params: EvaluationParams = {}

    def __init__(self, path: str):
        with open(path, 'r') as f:
            config_json = json.loads(f.read())
        self.execution_type = config_json['execution_type']
        self.state_mode = config_json['state_mode']
        self.cuda = config_json['cuda'] and torch.cuda.is_available()
        self.training_params = TrainingParams(config_json['training_params'])
        self.evaluation_params = EvaluationParams(config_json['evaluation_params'])

    def __str__(self):
        d = self.__dict__
        d['training_params'] = self.training_params.__dict__
        d['evaluation_params'] = self.evaluation_params.__dict__
        return json.dumps(d, indent=2)
