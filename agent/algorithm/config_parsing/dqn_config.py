import json
from pathlib import Path

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.environment_parameters import EnvironmentParameters


class DqnConfig:

    # ToDo: This still represents the wierd observation state modes (1 = OHLC, 2 = OHLC + trend, etc...)
    observation_space: int = 1
    window_size: int = 1
    episodes: int = 1
    batch_size: int = 1
    agent: AgentParameters
    environment: EnvironmentParameters

    def __init__(self, path: Path = None):
        if path is not None:
            with open(path, 'r') as f:
                config_json = json.loads(f.read())
            self.observation_space = config_json['observation_space']
            self.window_size = config_json['window_size']
            self.batch_size = config_json['batch_size']
            self.episodes = config_json['episodes']
            self.agent = AgentParameters(config_json['agent'])
            self.environment = EnvironmentParameters(config_json['environment'])

    @classmethod
    def from_dict(cls, dct: dict):
        inst = DqnConfig()
        inst.observation_space = dct['observation_space']
        inst.window_size = dct['window_size']
        inst.batch_size = dct['batch_size']
        inst.episodes = dct['episodes']
        inst.agent = AgentParameters(dct['agent'])
        inst.environment = dct['environment']
        return inst

    def __str__(self):
        d = self.__dict__
        d['agent'] = self.agent.__dict__
        d['environment'] = self.environment.__dict__
        return json.dumps(d, indent=2)