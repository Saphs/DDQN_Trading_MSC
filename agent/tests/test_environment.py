import unittest

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame

from agent.algorithm.environment import Environment


class TestEnvironment(unittest.TestCase):

    _mock_data = {
        'open':             [100, 100, 100, 100],
        'high':             [200, 200, 200, 200],
        'low':              [  0,   0,   0,   0],
        'close':            [100, 100, 100, 100],
        'open_norm':        [0.1, 0.1, 0.1, 0.1],
        'high_norm':        [0.2, 0.2, 0.2, 0.2],
        'low_norm':         [0.0, 0.0, 0.0, 0.0],
        'close_norm':       [0.1, 0.1, 0.1, 0.1],
        'trend':            [0.0, 0.0, 0.0, 0.0],
        '%body':            [0.1, 0.1, 0.1, 0.1],
        '%upper-shadow':    [0.1, 0.1, 0.1, 0.1],
        '%lower-shadow':    [0.1, 0.1, 0.1, 0.1],
    }

    _action_name = "action_name"
    _realistic_data_frame = pd.read_csv("resources/dummy_google_processed.csv", index_col="Date")

    def test_state_mode_1(self):
        df: DataFrame = DataFrame(self._mock_data.copy())
        env: Environment = Environment(data=df, state_mode=1)
        self.assertEqual(env.state_size, 4)

    def test_state_mode_1_data_preparation(self):
        df: DataFrame = self._realistic_data_frame.copy()
        env: Environment = Environment(data=df, state_mode=1)
        expected_states = torch.tensor([
            [0.9985537710358678, 0.9999999999999996, 1.0, 0.9999999999999996],
            [1.0, 0.9890173599598948, 0.9835606729254684, 0.982159061487112],
            [0.991699888041384, 0.9759965129737478, 0.891134993708016, 0.8804785058793563],
            [0.8881972366237103, 0.8716953194913697, 0.8076594146028366, 0.7889463822956073],
            [0.7787838944680017, 0.8273052298600629, 0.7861057480827554, 0.8401421801051923]
        ])
        self.assertIsNone(np.testing.assert_array_equal(env.states[0:5], expected_states))

    def test_state_mode_2(self):
        df: DataFrame = DataFrame(self._mock_data.copy())
        env: Environment = Environment(data=df, state_mode=2)
        self.assertEqual(env.state_size, 5)

    def test_state_mode_2_data_preparation(self):
        df: DataFrame = self._realistic_data_frame.copy()
        env: Environment = Environment(data=df, state_mode=2)
        expected_states = torch.tensor([
            [0.9985537710358678, 0.9999999999999996, 1.0, 0.9999999999999996, 0.0],
            [1.0, 0.9890173599598948, 0.9835606729254684, 0.982159061487112, 0.0],
            [0.991699888041384, 0.9759965129737478, 0.891134993708016, 0.8804785058793563, 0.0],
            [0.8881972366237103, 0.8716953194913697, 0.8076594146028366, 0.7889463822956073, 0.0],
            [0.7787838944680017, 0.8273052298600629, 0.7861057480827554, 0.8401421801051923, 0.0]
        ])
        self.assertIsNone(np.testing.assert_array_equal(env.states[0:5], expected_states))

    def test_state_mode_3(self):
        df: DataFrame = DataFrame(self._mock_data.copy())
        env: Environment = Environment(data=df, state_mode=3)
        self.assertEqual(env.state_size, 8)

    def test_state_mode_3_data_preparation(self):
        df: DataFrame = self._realistic_data_frame.copy()
        env: Environment = Environment(data=df, state_mode=3)
        expected_states = torch.tensor([
            [0.9985537710358678, 0.9999999999999996, 1.0, 0.9999999999999996, 0.0, 0.0379499202694547, 0.4857686851170349, 0.4762813946135103],
            [1.0, 0.9890173599598948, 0.9835606729254684, 0.982159061487112, 0.0, 0.5063490739539669, 0.1047597995266191, 0.3888911265194139],
            [0.991699888041384, 0.9759965129737478, 0.891134993708016, 0.8804785058793563, 0.0, 0.902563442607132, 0.0, 0.0974365573928679],
            [0.8881972366237103, 0.8716953194913697, 0.8076594146028366, 0.7889463822956073, 0.0, 0.8818438255187754, 0.0345812866311953, 0.0835748878500292],
            [0.7787838944680017, 0.8273052298600629, 0.7861057480827554, 0.8401421801051923, 0.0, 0.7086270099548022, 0.0869893220190746,0.2043836680261231]
        ])
        self.assertIsNone(np.testing.assert_array_equal(env.states[0:5], expected_states))

    def test_state_mode_4(self):
        df: DataFrame = DataFrame(self._mock_data.copy())
        env: Environment = Environment(data=df, state_mode=4)
        self.assertEqual(env.state_size, 3)

    def test_state_mode_4_data_preparation(self):
        df: DataFrame = self._realistic_data_frame.copy()
        env: Environment = Environment(data=df, state_mode=4)
        expected_states = torch.tensor([
            [0.0379499202694547, 0.4857686851170349, 0.4762813946135103],
            [0.5063490739539669, 0.1047597995266191, 0.3888911265194139],
            [0.902563442607132, 0.0, 0.0974365573928679],
            [0.8818438255187754, 0.0345812866311953, 0.0835748878500292],
            [0.7086270099548022, 0.0869893220190746, 0.2043836680261231]
        ])
        self.assertIsNone(np.testing.assert_array_equal(env.states[0:5], expected_states))

    def test_state_mode_5(self):
        df: DataFrame = DataFrame(self._mock_data.copy())
        env: Environment = Environment(data=df, state_mode=5, window_size=1)
        self.assertEqual(env.state_size % 4, 0)
        self.assertEqual(env.state_size, 4)

        env: Environment = Environment(data=df, state_mode=5, window_size=2)
        self.assertEqual(env.state_size % 4, 0)
        self.assertEqual(env.state_size, 8)

        env: Environment = Environment(data=df, state_mode=5, window_size=3)
        self.assertEqual(env.state_size % 4, 0)
        self.assertEqual(env.state_size, 12)

    def test_state_mode_5_windows_size_1_data_preparation(self):
        df: DataFrame = self._realistic_data_frame.copy()
        env: Environment = Environment(data=df, state_mode=5, window_size=1)
        expected_states = torch.tensor([
            [0.9985537710358678, 0.9999999999999996, 1.0, 0.9999999999999996],
            [1.0, 0.9890173599598948, 0.9835606729254684, 0.982159061487112],
            [0.991699888041384, 0.9759965129737478, 0.891134993708016, 0.8804785058793563],
            [0.8881972366237103, 0.8716953194913697, 0.8076594146028366, 0.7889463822956073],
            [0.7787838944680017, 0.8273052298600629, 0.7861057480827554, 0.8401421801051923]
        ])
        self.assertIsNone(np.testing.assert_array_equal(env.states[0:5], expected_states))

    def test_state_mode_5_windows_size_2_data_preparation(self):
        df: DataFrame = self._realistic_data_frame.copy()
        env: Environment = Environment(data=df, state_mode=5, window_size=2)
        expected_states = torch.tensor([
            [0.9985537710358678, 0.9999999999999996, 1.0, 0.9999999999999996, 1.0, 0.9890173599598948, 0.9835606729254684, 0.982159061487112],
            [1.0, 0.9890173599598948, 0.9835606729254684, 0.982159061487112, 0.991699888041384, 0.9759965129737478, 0.891134993708016, 0.8804785058793563],
            [0.991699888041384, 0.9759965129737478, 0.891134993708016, 0.8804785058793563, 0.8881972366237103, 0.8716953194913697, 0.8076594146028366, 0.7889463822956073],
            [0.8881972366237103, 0.8716953194913697, 0.8076594146028366, 0.7889463822956073, 0.7787838944680017, 0.8273052298600629, 0.7861057480827554, 0.8401421801051923],
            [0.7787838944680017, 0.8273052298600629, 0.7861057480827554, 0.8401421801051923, 0.8571339064267294,0.8352624228890746,0.816122800300902,0.8342599623190377]
        ])
        self.assertIsNone(np.testing.assert_array_equal(env.states[0:5], expected_states))

    def test_state_mode_5_windows_size_3_data_preparation(self):
        df: DataFrame = self._realistic_data_frame.copy()
        env: Environment = Environment(data=df, state_mode=5, window_size=3)
        expected_states = torch.tensor([
            [
                0.9985537710358678,
                0.9999999999999996,
                1.0,
                0.9999999999999996,

                1.0,
                0.9890173599598948,
                0.9835606729254684,
                0.982159061487112,

                0.991699888041384,
                0.9759965129737478,
                0.891134993708016,
                0.8804785058793563
            ],
            [
                1.0,
                0.9890173599598948,
                0.9835606729254684,
                0.982159061487112,

                0.991699888041384,
                0.9759965129737478,
                0.891134993708016,
                0.8804785058793563,

                0.8881972366237103,
                0.8716953194913697,
                0.8076594146028366,
                0.7889463822956073
            ],
            [
                0.991699888041384,
                0.9759965129737478,
                0.891134993708016,
                0.8804785058793563,

                0.8881972366237103,
                0.8716953194913697,
                0.8076594146028366,
                0.7889463822956073,

                0.7787838944680017,
                0.8273052298600629,
                0.7861057480827554,
                0.8401421801051923
            ],
            [
                0.8881972366237103,
                0.8716953194913697,
                0.8076594146028366,
                0.7889463822956073,

                0.7787838944680017,
                0.8273052298600629,
                0.7861057480827554,
                0.8401421801051923,

                0.8571339064267294,
                0.8352624228890746,
                0.816122800300902,
                0.8342599623190377
            ],
            [
                0.7787838944680017,
                0.8273052298600629,
                0.7861057480827554,
                0.8401421801051923,

                0.8571339064267294,
                0.8352624228890746,
                0.816122800300902,
                0.8342599623190377,

                0.8143116864388862,
                0.7938316226362327,
                0.7793471340897908,
                0.7655460314974425
            ]
        ])
        self.assertIsNone(np.testing.assert_array_equal(env.states[0:5], expected_states))


if __name__ == '__main__':
    unittest.main()
