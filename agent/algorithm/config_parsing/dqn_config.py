from dataclasses import dataclass

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.environment_parameters import EnvironmentParameters


@dataclass
class DqnConfig:
    """Data class representing all available configuration values."""
    seed: int = 1
    check_points: int = 1
    # ToDo: This still represents the wierd observation state modes (1 = OHLC, 2 = OHLC + trend, etc...)
    observation_space: int = 1
    window_size: int = 1
    episodes: int = 1
    batch_size: int = 1
    reward_function: str = ""
    agent: AgentParameters = AgentParameters()
    environment: EnvironmentParameters = EnvironmentParameters()
