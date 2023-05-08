from dataclasses import dataclass
from datetime import datetime


@dataclass
class Stock:
    """Class for storing meta information about an available stock."""
    db_id: int
    name: str
    symbol: str
    size: int
    first_date: datetime
    last_date: datetime