
class AgentParameters:
    n_step: int
    gamma: int
    replay_memory_size: int
    target_net_update_interval: int


    def __init__(self, config: dict):
        self.n_step = config['n_step']
        self.gamma = config['gamma']
        self.replay_memory_size = config['replay_memory_size']
        self.target_net_update_interval = config['target_net_update_interval']
