from dataclasses import dataclass


@dataclass
class EnvironmentParameters:
    data_set_name: str = ''
    density: int = 0
    step_size: int = 0
    look_ahead: int = 0
    transaction_cost: float = 0.0
    initial_capital: float = 0
