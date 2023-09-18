from dataclasses import dataclass


@dataclass
class EnvironmentParameters:
    data_set_name: str = ''
    density: int = 0
    stride: int = 0
    transaction_cost: float = 0.0
    initial_capital: float = 0
