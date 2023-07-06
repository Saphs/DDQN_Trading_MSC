from dataclasses import dataclass
from datetime import datetime
from typing import List

from dqn.algorithm.config_parsing.agent_parameters import AgentParameters
from dqn.algorithm.config_parsing.dqn_config import DqnConfig
from dqn.algorithm.config_parsing.environment_parameters import EnvironmentParameters


def custom_encode(o):
    return o.isoformat() if (isinstance(o, datetime)) else o.__dict__

def custom_decode(dct):
    # ToDo: All this dataclass and decoding stuff feels very flaky - I should address this
    if 'gamma' in dct and 'target_net_update_interval' in dct:
        return AgentParameters(dct)
    elif 'data_set_name' in dct:
        return EnvironmentParameters(dct)
    elif 'agent' in dct and 'environment' in dct:
        return DqnConfig.from_dict(dct)
    elif 't_start' in dct and 'config' in dct:
        return TrainingSession(
            datetime.fromisoformat(dct['t_start']),
            datetime.fromisoformat(dct['t_end']),
            dct['config']
        )
    elif 'total_episodes' in dct:
        return History(dct['agent_name'], dct['total_episodes'], dct['sessions'])
    else:
        return dct


@dataclass
class TrainingSession:
    """Data class containing all information relevant to a single training session."""
    t_start: datetime
    t_end: datetime
    config: DqnConfig


@dataclass
class History:
    """Data class containing a record of all trainings performed on a given model."""
    agent_name: str
    total_episodes: int
    sessions: List[TrainingSession]

    def append_session(self, s: TrainingSession) -> None:
        self.total_episodes += s.config.episodes
        self.sessions.append(s)
