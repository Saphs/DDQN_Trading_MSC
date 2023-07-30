import unittest

import testing_utilities
from agent.Evaluation import Evaluation
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.environment import Environment


class TestEvaluation(unittest.TestCase):

    def test_prediction(self):
        test_config: DqnConfig = testing_utilities.get_test_config()
        print(test_config)
        env_data = testing_utilities.get_test_stock_data()
        environment: Environment = Environment.init(env_data, config=test_config)
        evaluation = Evaluation(None, environment, "testing_dummy", environment.window_size, "Unittest")

        predicted_df = evaluation._predict()
        print(predicted_df)



    def test_calculate_effective_transactions(self):
        test_config: DqnConfig = testing_utilities.get_test_config()
        env_data = testing_utilities.get_test_stock_data()
        environment = Environment.init(env_data, config=test_config)
        evaluation = Evaluation(None, environment, "testing_dummy", environment.window_size, "Unittest")

        transactions = evaluation.raw_data[['prediction', 'lact_action', 'effective_transaction']]
        print(transactions)
