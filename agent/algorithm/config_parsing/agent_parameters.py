from dataclasses import dataclass


@dataclass
class AgentParameters:
    style: str = ''
    alpha: float = 0.0
    gamma: float = 0.0
    epsilon_start: float = 0.0
    epsilon_end: float = 0.0
    epsilon_decay: float = 0
    replay_memory_size: int = 0
    target_net_update_interval: int = 0

