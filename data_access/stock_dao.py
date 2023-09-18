import json
from typing import List, Tuple

import pandas as pd
import psycopg2

from pandas import DataFrame

from data_access.stock import Stock



class StockDao:

    _DB_DATE_FORMAT = "%Y-%m-%d"

    def __init__(self, path):
        with open(path, 'rb') as db_file:
            self._DB_CONFIG = json.load(db_file)

        self.DB_CONNECTION = psycopg2.connect(
            database=self._DB_CONFIG['database'],
            user=self._DB_CONFIG['user'],
            password=self._DB_CONFIG['password'],
            host=self._DB_CONFIG['host'],
            port=self._DB_CONFIG['port']
        )

        self.DB_CONNECTION.autocommit = True
        self.db_cursor = self.DB_CONNECTION.cursor()

    def list_available(self) -> List[Stock]:
        data_point_query = "SELECT COUNT(date), MIN(date), MAX(date) FROM stock_data.data_points WHERE stock_id = %s;"
        self.db_cursor.execute("SELECT * FROM stock_data.stocks;")
        known_entries = self.db_cursor.fetchall()
        stocks = []
        for sym, _id, name in known_entries:
            self.db_cursor.execute(data_point_query, [_id])
            cnt, _min, _max = self.db_cursor.fetchall()[0]
            stocks.append(Stock(_id, name, sym, cnt, _min, _max))
        return stocks

    def get_stock_meta(self, name: str) -> Stock:
        self.db_cursor.execute("SELECT * FROM stock_data.stocks WHERE name = %s;", [name])
        data_point_query = "SELECT COUNT(date), MIN(date), MAX(date) FROM stock_data.data_points WHERE stock_id = %s;"
        sym, _id, name = self.db_cursor.fetchone()
        self.db_cursor.execute(data_point_query, [_id])
        cnt, _min, _max = self.db_cursor.fetchone()

        return Stock(_id, name, sym, cnt, _min, _max)

    def get_stock_data(self, name: str) -> DataFrame:
        stock = self.get_stock_meta(name)
        self.db_cursor.execute("SELECT open, high, low,  close, adjusted_close, volume, date FROM stock_data.data_points WHERE stock_id = %s;", [stock.db_id])
        column_names = list(map(lambda c: c.name, self.db_cursor.description))
        data_points = DataFrame(data=self.db_cursor.fetchall(), columns=column_names)
        data_points['date'] = pd.to_datetime(data_points['date'],  format=self._DB_DATE_FORMAT)

        return data_points
