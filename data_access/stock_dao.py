import json
from typing import List, Tuple

import pandas as pd
import psycopg2

from pandas import DataFrame

from data_access.stock import Stock

with open("db_config.json", 'rb') as db_file:
    _DB_CONFIG = json.load(db_file)

class StockDao:

    _DB_DATE_FORMAT = "%Y-%m-%d"

    def __init__(self):
        self.DB_CONNECTION = psycopg2.connect(
            database=_DB_CONFIG['database'],
            user=_DB_CONFIG['user'],
            password=_DB_CONFIG['password'],
            host=_DB_CONFIG['host'],
            port=_DB_CONFIG['port']
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

    def get_stock(self, name: str) -> Tuple[Stock, DataFrame]:
        self.db_cursor.execute("SELECT * FROM stock_data.stocks WHERE name = %s;", [name])
        data_point_query = "SELECT COUNT(date), MIN(date), MAX(date) FROM stock_data.data_points WHERE stock_id = %s;"
        sym, _id, name = self.db_cursor.fetchone()
        self.db_cursor.execute(data_point_query, [_id])
        cnt, _min, _max = self.db_cursor.fetchone()
        self.db_cursor.execute("SELECT open, high, low,  close, adjusted_close, volume, date FROM stock_data.data_points WHERE stock_id = %s;", [_id])

        column_names = list(map(lambda c: c.name, self.db_cursor.description))
        data_points = DataFrame(data=self.db_cursor.fetchall(), columns=column_names)
        data_points['date'] = pd.to_datetime(data_points['date'],  format=self._DB_DATE_FORMAT)

        return Stock(_id, name, sym, cnt, _min, _max), data_points