import logging
import math
import typing
import warnings
import pandas as pd
from pandas import DataFrame
import hashlib
from sklearn.preprocessing import MinMaxScaler

import os
from pathlib import Path

from data_access.stock import Stock
from data_access.stock_dao import StockDao
from data_access.pattern_detection.label_candles import label_candles


def _cache_name(stock: Stock) -> str:
    """Generate deterministic unique name for given stock."""
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
    """ DatasetLoader to access data from the database"""

    def __init__(self, cache_dir: Path, dataset_name: str, skip_cache: bool = False, db_config: str = "db_config.json"):
        self._cache_dir = cache_dir
        self._stock_dao = StockDao(db_config)

        if not self._is_stock_known(dataset_name):
            raise ValueError(f"Unknown dataset name: {dataset_name}. Ensure the stock selected is known.")
        else:
            self.stock, self.data = self._stock_from_db(dataset_name, skip_cache)

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

    def _stock_from_db(self, stock_name: str, skip_cache: bool) -> typing.Tuple[Stock, DataFrame]:
        """Load data from data base"""
        logging.info(f"Retrieving stock {stock_name}")
        stock = self._stock_dao.get_stock_meta(stock_name)

        if not self._has_cache_file(stock) or skip_cache:
            db_data = self._stock_dao.get_stock_data(stock_name)
            db_data.set_index('date', inplace=True)
            db_data['adjusted_close'] = db_data['close']
            db_data.drop(['adjusted_close', 'volume'], axis=1, inplace=True)
            logging.info(
                f"Labeling candles with technical indicators ({len(db_data)} candles) this can take some time."
            )
            db_data = db_data.dropna(axis=0)
            label_candles(db_data)
            db_data = _add_normalized_data(db_data)
            self._save_cache_file(stock, db_data)
        else:
            db_data = self._load_cache_file(stock)

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

    def _load_cache_file(self, stock: Stock) -> DataFrame:
        cache_file = os.path.join(self._cache_dir, f"{_cache_name(stock)}.csv")
        logging.info(f"Loading cache entry: {cache_file}")
        return pd.read_csv(cache_file, index_col='date')


