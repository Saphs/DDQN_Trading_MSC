
class TrainingParams:
    data_set_name: str
    data_set_base_path: str
    result_base_path: str
    episodes: int
    n_step: int
    gamma: int
    replay_memory_size: int
    batch_size: int
    transaction_cost: float
    target_net_update_interval: int
    window_size: int

    def __init__(self, config: dict):
        self.data_set_name = config['data_set_name']
        self.data_set_base_path = config['data_set_base_path']
        self.result_base_path = config['result_base_path']
        self.episodes = config['episodes']
        self.n_step = config['n_step']
        self.gamma = config['gamma']
        self.replay_memory_size = config['replay_memory_size']
        self.batch_size = config['batch_size']
        self.transaction_cost = config['transaction_cost']
        self.target_net_update_interval = config['target_net_update_interval']
        self.window_size = config['window_size']
