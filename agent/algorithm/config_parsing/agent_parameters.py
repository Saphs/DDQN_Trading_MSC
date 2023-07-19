
class AgentParameters:
    style: str
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    replay_memory_size: int
    target_net_update_interval: int

    def __init__(self, config: dict):
        self.style = config['style']  # ToDo: Running train on the latest model causes issues here
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.replay_memory_size = config['replay_memory_size']
        self.target_net_update_interval = config['target_net_update_interval']
