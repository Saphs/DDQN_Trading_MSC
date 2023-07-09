
class EnvironmentParameters:
    data_set_name: str
    transaction_cost: float
    initial_capital: float

    def __init__(self, config: dict):
        self.data_set_name = config['data_set_name']
        self.transaction_cost = config['transaction_cost']
        self.initial_capital = config['initial_capital']
