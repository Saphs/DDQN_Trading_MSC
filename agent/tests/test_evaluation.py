import unittest

import testing_utilities
from agent.Evaluation import Evaluation
from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.config_parsing.environment_parameters import EnvironmentParameters
from agent.algorithm.environment import Environment


class TestEvaluation(unittest.TestCase):

    def test_prediction(self):
        _config = DqnConfig(
            observation_space=5,
            window_size=3,
            episodes=1,
            batch_size=1,
            agent=AgentParameters(
                style='dqn',
                alpha=0.001,
                gamma=0.9,
                epsilon_start=1.0,
                epsilon_end=0.0,
                epsilon_decay=100,
                replay_memory_size=1,
                target_net_update_interval=1
            ),
            environment=EnvironmentParameters(
                data_set_name='Google',
                stride=0,
                transaction_cost=0.0,
                initial_capital=1000.0
            )
        )

        env_data = testing_utilities.get_test_stock_data()
        environment: Environment = Environment(env_data, config=_config)
        evaluation = Evaluation(None, environment, "testing_dummy", environment.window_size, "Unittest")

        predicted_df = evaluation._predict()
        print(env_data)
        print(predicted_df)



    def test_calculate_effective_transactions(self):
        test_config: DqnConfig = testing_utilities.get_test_config()
        env_data = testing_utilities.get_test_stock_data()
        environment = Environment(env_data, config=test_config)
        evaluation = Evaluation(None, environment, "testing_dummy", environment.window_size, "Unittest")

        transactions = evaluation.raw_data[['prediction', 'lact_action', 'effective_transaction']]
        print(transactions)
