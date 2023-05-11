from typing import List

import pandas as pd
from scipy import stats


def arithmetic_mean(values: pd.Series) -> float:
    """The simplest average of a list of numbers"""
    return values.sum() / len(values)


def geometric_mean(values: pd.Series) -> float:
    """
    The geometric mean averages the distance of factors
    See: https://de.wikipedia.org/wiki/Geometrisches_Mittel
    """
    return float(stats.gmean(values.values))
