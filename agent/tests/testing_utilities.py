import os.path
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.config_parsing.dqn_config_codec import DqnConfigCodec

_TEST_STOCK_PATH = Path("./resources/dummy_google_processed.csv")
_TEST_CONFIG_PATH = Path("./resources/dummy_config.json")


def get_test_stock_data() -> DataFrame:
    return pd.read_csv(_TEST_STOCK_PATH)


def get_test_config() -> DqnConfig:
    return DqnConfigCodec.read_json(_TEST_CONFIG_PATH)
