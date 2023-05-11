class EvaluationParams:
    data_set_name: str
    data_set_base_path: str
    model_base_path: str
    model_folder_name: str
    n_step: int
    batch_size: int
    window_size: int
    transaction_cost: int

    def __init__(self, config: dict):
        self.data_set_name = config['data_set_name']
        self.data_set_base_path = config['data_set_base_path']
        self.model_base_path = config['model_base_path']
        self.model_folder_name = config['model_folder_name']
        self.n_step = config['n_step']
        self.batch_size = config['batch_size']
        self.window_size = config['window_size']
        self.transaction_cost = config['transaction_cost']
