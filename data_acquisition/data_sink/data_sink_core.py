import uuid
from typing import Tuple, List, Optional
import click
import pandas as pd
import logging
import psycopg2
from pandas import DataFrame
from psycopg2.extras import execute_values
import plotly.graph_objects as go

_RUN_DRY = False
_DB_CONFIG = {
    "database": "ML_Stock_Data",
    "user": "postgres",
    "password": "",
    "host": "127.0.0.1",
    "port": "5432"
}
_CYAN = '\033[96m'
_CLR_RST = '\033[0m'
_SUPPORTED_FILE_ENDINGS = dict(
        csv="csv"
)
# Order is important
_EXPECTED_KEYS = [
    "high",
    "low",
    "open",
    "close",
    "adjusted_close",
    "volume",
    "date",
]

def _ensure_valid_name(preferred_name: str, file_name: str) -> str:
    if preferred_name is None:
        generated_name = f"auto_{file_name.split('/')[-1].replace('.','_')}_{str(uuid.uuid1())[:8]}"
        logging.info(f"No preferred name was provided. Generated: '{generated_name}' as a replacement.")
        return generated_name
    return preferred_name

class DataSink:

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

    def import_from_file(self, source_file: click.File, name: str, symbol: str = "UNKNOWN"):
        _name = _ensure_valid_name(name, source_file.name)
        file_ending = source_file.name.split('.')[-1]
        if file_ending == _SUPPORTED_FILE_ENDINGS['csv']:
            df = pd.read_csv(source_file)
            # Key maps should be defined like this: { "original_name": "new_name" }
            df = self._prepare_for_db(df, {"Adj Close": "adjusted_close"})
            meta_entry_id = self._add_meta_entry(_name, symbol)
            df["stock_id"] = meta_entry_id
            self._bulk_insert(df)
        else:
            logging.error(f"File type not supported. Supported types are: {list(_SUPPORTED_FILE_ENDINGS.values())}")

    def list(self) -> str:
        data_point_query = "SELECT COUNT(date), MIN(date), MAX(date) FROM stock_data.data_points WHERE stock_id = %s;"
        self.db_cursor.execute("SELECT * FROM stock_data.stocks;")
        known_stocks = self.db_cursor.fetchall()
        smry: str = "+-Known data sets---------------------\n"
        empty = 0
        for sym, _id, name in known_stocks:
            self.db_cursor.execute(data_point_query, [_id])
            cnt, _min, _max = self.db_cursor.fetchall()[0]
            if cnt == 0:
                empty = empty + 1
            smry = smry + (f"| {sym}\t({_CYAN}{name}{_CLR_RST}, {_id=})\t-> entries: {cnt} from: {_min} to: {_max}\n")
        smry = smry + "+-Cleanliness-------------------------\n"
        smry = smry + f"| Empty data sets: {empty}\n"
        return smry + "+-------------------------------------"

    def clean(self) -> None:
        data_point_query = "SELECT COUNT(id) FROM stock_data.data_points WHERE stock_id = %s;"
        self.db_cursor.execute("SELECT * FROM stock_data.stocks;")
        known_stocks = self.db_cursor.fetchall()
        for sym, _id, name in known_stocks:
            self.db_cursor.execute(data_point_query, [_id])
            cnt = self.db_cursor.fetchall()[0][0]
            if cnt == 0:
                self.db_cursor.execute("DELETE FROM stock_data.stocks WHERE stock_id = %s", [_id])

    def plot(self, name: str, identifier: int) -> None:
        if name is not None:
            self._plot_by_name(name)
        if identifier is not None:
            self._plot_by_id(identifier)
        if name is None and identifier is None:
            logging.info("At least one identifier needs to be provided (name or identifier number).")

    def _plot_by_name(self, name: str):
        self.db_cursor.execute("SELECT * FROM stock_data.stocks WHERE name = %s", [name])
        fetched = self.db_cursor.fetchall()
        if len(fetched) <= 0:
            logging.warning(f"Name: '{name}' could not be found.")
        else:
            _, idx, _ = fetched[0]
            self._plot_by_id(idx)
        pass

    def _plot_by_id(self, identifier: int):
        self.db_cursor.execute("SELECT * FROM stock_data.stocks WHERE stock_id = %s", [identifier])
        sym, idx, name = self.db_cursor.fetchone()
        self.db_cursor.execute("SELECT * FROM stock_data.data_points WHERE stock_id = %s", [identifier])
        df = DataFrame(self.db_cursor.fetchall())
        fig = go.Figure(data=[
            go.Candlestick(
                x=df[7],
                open=df[3],
                high=df[1],
                low=df[2],
                close=df[4]
            )
        ])
        fig.show()

    def rename(self, old: str, new: str):
        if len(new.strip()) >= 3:
            self.db_cursor.execute("SELECT * FROM stock_data.stocks WHERE stock_data.stocks.name = %s", [old])
            targets = self.db_cursor.fetchall()
            if len(targets) <= 0:
                logging.warning(f"Could not find old name: {old} in the known stocks.")
                return
            statement = "UPDATE stock_data.stocks SET name = %s WHERE stock_data.stocks.name = %s;"
            self.db_cursor.execute(statement, (new, old))
        else:
            logging.warning(f"Failed to update stock name. Name: {new=} must have at least 3 characters")

    def _prepare_for_db(self, df: DataFrame, key_map: Optional[dict] = None) -> DataFrame:
        """Validate and prepare the data coming in as a Dataframe before inserting it into the DB."""
        if key_map is not None:
            df = df.rename(columns=key_map)
        df.columns = map(str.lower, df.columns)

        if not df.shape[1] == len(_EXPECTED_KEYS):
            raise AssertionError(
                f"Found unexpected number of input columns (expected {len(_EXPECTED_KEYS)}, found: {df.shape[1]})"
            )
        if not set(_EXPECTED_KEYS).issubset(df.columns):
            raise AssertionError(
                f"Found unexpected column names\n\texpected: {_EXPECTED_KEYS}\n\tfound: {list(df.columns)}\n\t"
                f"provided translations: {key_map}"
            )
        return df

    def _bulk_insert(self, df: DataFrame):
        values = []
        for _, row in df.iterrows():
            entry = []
            for k in _EXPECTED_KEYS + ["stock_id"]:
                entry.append(row[k])
            values.append(tuple(entry))
        self._insert_values(values)

    def _insert_values(self, values: List[Tuple[float, float, float, float, float, float, str, int]]) -> None:
        if not _RUN_DRY:
            keys = ", ".join(_EXPECTED_KEYS + ["stock_id"])
            statement = f"INSERT INTO stock_data.data_points({keys}) VALUES %s"
            execute_values(self.db_cursor, statement, values)
            logging.info(f"Inserted {len(values)} rows")
        else:
            logging.info(f"Insert of {len(values)} rows was stopped via hard coded flag {_RUN_DRY=}")

    def _add_meta_entry(self, name: str, symbol: str) -> int:
        """Insert stock meta data into the meta data table and return the generated unique id."""
        statement = f"INSERT INTO stock_data.stocks(name, symbol) VALUES (%s, %s) RETURNING stock_id;"
        self.db_cursor.execute(statement, (name, symbol))
        unique_id = self.db_cursor.fetchone()[0]
        logging.info(f"Insert of {(name, symbol)} resulted in entry id: {unique_id}")
        return unique_id

