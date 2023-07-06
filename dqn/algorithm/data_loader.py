import logging
import math
import typing
import warnings
import pandas as pd
from pandas import DataFrame
import hashlib
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path

from data_access.stock import Stock
from data_access.stock_dao import StockDao
from dqn.algorithm.pattern_detection.label_candles import label_candles


def _cache_name(stock: Stock) -> str:
    return f"{stock.symbol}_" + hashlib.md5(f"{stock.name}-{stock.symbol}-{stock.size}-{stock.db_id}".encode("UTF-8")).hexdigest()

def _add_normalized_data(df: DataFrame) -> DataFrame:
    """Represent OHLC as values between 1.0 and -1.0"""
    min_max_scaler = MinMaxScaler()
    df['open_norm'] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
    df['high_norm'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low_norm'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['close_norm'] = min_max_scaler.fit_transform(df.close.values.reshape(-1, 1))
    return df

class DataLoader:
    """ Dataset form GOOGLE"""

    def __init__(self, cache_dir: Path, dataset_name: str):
        warnings.filterwarnings('ignore')
        self._cache_dir = cache_dir
        self._stock_dao = StockDao()

        if not self._is_stock_known(dataset_name):
            raise ValueError(f"Unknown dataset name: {dataset_name}. Ensure the stock selected is known.")
        else:
            self.stock, self.data = self._stock_from_db(dataset_name)

    def get(self) -> DataFrame:
        return self.data

    def get_section(self,  split_point: float, begin_date=None, end_date=None) -> typing.Tuple[DataFrame, DataFrame]:
        if 0 <= split_point <= 1.0:
            dates = self.data.reset_index()['date']
            split_date = dates.iloc[math.floor((len(dates) * split_point))]
            return self.get_section_by_date(str(split_date), begin_date, end_date)
        else:
            ValueError(f"Float: {split_point=} must be between 0.0 and 1.0.")

    def get_section_by_date(self,  split_point: str, begin_date=None, end_date=None) -> typing.Tuple[DataFrame, DataFrame]:
        tmp_data = self.data
        if begin_date is not None:
            tmp_data = tmp_data[tmp_data.index >= begin_date]
        if end_date is not None:
            tmp_data = tmp_data[tmp_data.index <= end_date]
        return tmp_data[tmp_data.index < split_point], tmp_data[tmp_data.index >= split_point]

    def _is_stock_known(self, dataset_name: str) -> bool:
        known_stocks: typing.List[str] = list(map(lambda st: st.name, self._stock_dao.list_available()))
        return dataset_name in known_stocks


    def _stock_from_db(self, stock_name: str) -> typing.Tuple[Stock, DataFrame]:
        """Load data from data base"""
        logging.info(f"Retrieving stock {stock_name}")

        # ToDo: This retrieves the full unprocessed data for the stock as well even though there might be a cache file including this
        stock, db_data = self._stock_dao.get_stock(stock_name)
        db_data.set_index('date', inplace=True)
        db_data['adjusted_close'] = db_data['close']
        db_data.drop(['adjusted_close', 'volume'], axis=1, inplace=True)

        if not self._has_cache_file(stock):
            logging.info(f"Labeling candles with technical indicators ({len(db_data)} candles) this can take some time.")
            label_candles(db_data)
            self._save_cache_file(stock, db_data)
        else:
            db_data = self._load_label_candles(stock)

        db_data = _add_normalized_data(db_data)
        return stock, db_data

    def _has_cache_file(self, stock: Stock) -> bool:
        if not os.path.exists(self._cache_dir):
            return False
        cache_name = f"{_cache_name(stock)}.csv"
        return cache_name in os.listdir(self._cache_dir)

    def _save_cache_file(self, stock: Stock, df: DataFrame) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = os.path.join(self._cache_dir, f"{_cache_name(stock)}.csv")
        df.to_csv(cache_file)
        logging.info(f"Saved cache file under: {cache_file}")

    def _load_label_candles(self, stock: Stock) -> DataFrame:
        cache_file = os.path.join(self._cache_dir, f"{_cache_name(stock)}.csv")
        logging.info(f"Loading cache entry: {cache_file}")
        return pd.read_csv(cache_file, index_col='date')

    def _plot_data(self):
        """
        INTERNAL TESTING FUNCTION.
        Used for debugging only - probably broken by now
        """
        sns.set(rc={'figure.figsize': (9, 5)})
        df1 = pd.Series(self.data_train.close, index=self.data.index)
        df2 = pd.Series(self.data_test.close, index=self.data.index)
        ax = df1.plot(color='b', label='Train')
        df2.plot(ax=ax, color='r', label='Test')
        ax.set(xlabel='Time', ylabel='Close Price')
        ax.set_title(f'Train and Test sections of dataset {self.DATA_NAME}')
        plt.legend()
        data_path = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,f'data_plot') + '/'
        plt.savefig(f'{Path(data_path).parent}/{self.DATA_NAME}.jpg', dpi=300)


