from typing import List, Set

import numpy as np
from pandas import DataFrame

from .technical_indicator_functions import (is_bearish_engulfing, is_bullish_engulfing, is_dark_cloud_cover,
                      is_doji, is_evening_star, is_falling_three_methods, is_hammer, is_hanging_man,
                      is_inverse_hammer, is_morning_star, is_piercing_line, is_rising_three_methods,
                      is_shooting_star, is_spinning_top, is_three_black_crows, is_three_white_soldier,
                      is_bullish_harami, is_bearish_harami)

from tqdm import tqdm

# Trend Parameters
window_size = 20

# Candle stick pattern parameters
gap_significance_level = 0.05
candle_significance_level = 0.1
doji_length_percentage = 0.01

# star parameters
down_gap_percentage_morning_star = 0.1
up_gap_percentage_evening_star = 0.1

# hammer parameters
percentage_of_shadow_hammer = 0.2
upper_bound_hammer_significance_level = 0.4
lower_bound_hammer_significance_level = 0.1

# hangman parameters
percentage_of_upper_shadow = 0.2
upper_bound_hangman_significance_level = 0.4
lower_bound_hangman_significance_level = 0.1

# spanning top parameters
spanning_top_significance = 0.1
spanning_top_doji_level = 0.3
spanning_top_offset = 0.1

# Harami parameters
harami_gap_significance_level = 0.05


def _list_of_empty_sets(size: int) -> List[Set]:
    return list(map(lambda x: set(), np.empty((size, 0))))

def label_candles(df: DataFrame) -> None:
    average_range_of_candles_bodies = abs(df.close - df.open).mean()

    # Initialize new column full of empty lists
    df['label'] = _list_of_empty_sets(len(df))
    df['action'] = "None"
    df['%body'] = abs(df.close - df.open) / (df.high - df.low)
    df['%upper-shadow'] = (df.high - df[['close', 'open']].max(axis=1)) / (df.high - df.low)
    df['%lower-shadow'] = (df[['close', 'open']].min(axis=1) - df.low) / (df.high - df.low)

    patterns = {"hammer": [], "inverse hammer": [], "bullish engulfing": [], "piercing line": [],
                "morning star": [], "three white soldiers": [], "hanging man": [], "shooting star": [],
                "bearish engulfing": [], "evening star": [], "three black crows": [], "dark cloud cover": [],
                "doji": [], "spanning top": [], "falling three methods": [], "rising three methods": [],
                "bullish harami": [], "bearish harami": []}

    find_trend(df, window_size)

    for i in tqdm(range(len(df) - 1), ncols=80):
        if is_hammer(df.iloc[i], percentage_of_upper_shadow=percentage_of_shadow_hammer,
                     upper_bound_hammer_significance_level=upper_bound_hammer_significance_level,
                     lower_bound_hammer_significance_level=lower_bound_hammer_significance_level):
            #patterns["hammer"].append(i)
            df['label'][i].add("hammer")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'up':
                df['action'][i + 1] = 'buy'

        if is_inverse_hammer(df.iloc[i], percentage_of_lower_shadow=percentage_of_shadow_hammer,
                             upper_bound_hammer_significance_level=upper_bound_hammer_significance_level,
                             lower_bound_hammer_significance_level=lower_bound_hammer_significance_level):
            #patterns["inverse hammer"].append(i)
            df['label'][i].add("inverse hammer")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'up':
                df['action'][i + 1] = 'buy'

        if (i > 0) and is_bullish_engulfing(df.iloc[i - 1], df.iloc[i], average_range_of_candles_bodies,
                                            candle_significance_level):
            #patterns["bullish engulfing"].append(i)
            df['label'][i].add("bullish engulfing")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'up':
                df['action'][i + 1] = 'buy'

        if (i > 0) and is_piercing_line(df.iloc[i - 1], df.iloc[i],
                                        average_range_of_candles_bodies,
                                        significance_level=candle_significance_level,
                                        gap_significance_level=gap_significance_level):
            #patterns["piercing line"].append(i)
            df['label'][i].add("piercing line")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'up':
                df['action'][i + 1] = 'buy'

        if (i > 1) and is_morning_star(df.iloc[i - 2], df.iloc[i - 1], df.iloc[i],
                                       average_range_of_candles_bodies,
                                       doji_length_percentage=doji_length_percentage,
                                       significance_level=candle_significance_level,
                                       down_percentage=down_gap_percentage_morning_star):
            #patterns["morning star"].append(i)
            df['label'][i].add("morning star")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'up':
                df['action'][i + 1] = 'buy'

        if (i > 1) and is_three_white_soldier(df.iloc[i - 2], df.iloc[i - 1], df.iloc[i],
                                              average_range_of_candles_bodies,
                                              candle_significance_level):
            #patterns["three white soldiers"].append(i)

            df['label'][i].add("three white soldiers")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'up':
                df['action'][i + 1] = 'buy'

        if (i > 0) and is_bullish_harami(df.iloc[i - 1], df.iloc[i], average_range_of_candles_bodies,
                                         candle_significance_level, harami_gap_significance_level):
            #patterns["bullish harami"].append(i)
            df['label'][i].add("bullish harami")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'up':
                df['action'][i + 1] = 'buy'

        if is_hanging_man(df.iloc[i], percentage_of_upper_shadow=percentage_of_upper_shadow,
                          lower_bound_hangman_significance_level=lower_bound_hangman_significance_level,
                          upper_bound_hangman_significance_level=upper_bound_hangman_significance_level):
            #patterns["hanging man"].append(i)
            df['label'][i].add("hanging man")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'down':
                df['action'][i + 1] = 'sell'

        if is_shooting_star(df.iloc[i], percentage_of_lower_shadow=percentage_of_upper_shadow,
                            lower_bound_hangman_significance_level=lower_bound_hangman_significance_level,
                            upper_bound_hangman_significance_level=upper_bound_hangman_significance_level):
            #patterns["shooting star"].append(i)
            df['label'][i].add("shooting star")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'down':
                df['action'][i + 1] = 'sell'

        if (i > 0) and is_bearish_engulfing(df.iloc[i - 1], df.iloc[i], average_range_of_candles_bodies,
                                            candle_significance_level):
            #patterns["bearish engulfing"].append(i)
            df['label'][i].add("bearish engulfing")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'down':
                df['action'][i + 1] = 'sell'

        if (i > 1) and is_evening_star(df.iloc[i - 2], df.iloc[i - 1], df.iloc[i],
                                       average_range_of_candles_bodies,
                                       doji_length_percentage=doji_length_percentage,
                                       significance_level=candle_significance_level,
                                       up_percentage=up_gap_percentage_evening_star):
            #patterns["evening star"].append(i)
            df['label'][i].add("evening star")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'down':
                df['action'][i + 1] = 'sell'

        if (i > 1) and is_three_black_crows(df.iloc[i - 2], df.iloc[i - 1], df.iloc[i],
                                            average_range_of_candles_bodies,
                                            significance_level=candle_significance_level):
            #patterns["three black crows"].append(i)

            df['label'][i].add("three black crows")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'down':
                df['action'][i + 1] = 'sell'

        if (i > 0) and is_dark_cloud_cover(df.iloc[i - 1], df.iloc[i],
                                           average_range_of_candles_bodies,
                                           significance_level=candle_significance_level,
                                           gap_significance_level=gap_significance_level):
            #patterns["dark cloud cover"].append(i)
            df['label'][i].add("dark cloud cover")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'down':
                df['action'][i + 1] = 'sell'

        if (i > 0) and is_bearish_harami(df.iloc[i - 1], df.iloc[i], average_range_of_candles_bodies,
                                         candle_significance_level, harami_gap_significance_level):
            #patterns["bearish harami"].append(i)
            df['label'][i].add("bearish harami")
            trend_history = df['trend'][i]
            if trend_history and confirmation_of_the_trend(df, i) == 'down':
                df['action'][i + 1] = 'sell'

        if is_doji(df.iloc[i], average_range_of_candles_bodies, percentage_length=doji_length_percentage):
            #patterns["doji"].append(i)
            df['label'][i].add("doji")

        if is_spinning_top(df.iloc[i], average_range_of_candles_bodies, significance_level=spanning_top_significance,
                           doji_level=spanning_top_doji_level,
                           offset=spanning_top_offset):
            #patterns["spanning top"].append(i)
            df['label'][i].add("spanning top")

        if (i > 3) and is_falling_three_methods(df.iloc[i - 4], df.iloc[i - 3], df.iloc[i - 2],
                                                df.iloc[i - 1], df.iloc[i], average_range_of_candles_bodies,
                                                significance_level=candle_significance_level):
            #patterns["falling three methods"].append(i)
            df['label'][i].add("falling three methods")

        if (i > 3) and is_rising_three_methods(df.iloc[i - 4], df.iloc[i - 3], df.iloc[i - 2],
                                               df.iloc[i - 1], df.iloc[i], average_range_of_candles_bodies,
                                               significance_level=candle_significance_level):
            #patterns["rising three methods"].append(i)
            df['label'][i].add("rising three methods")

    for i in range(len(df)):
        df['label'][i] = list(df['label'][i])



def find_trend(data, window_size=20):
    data['sm_avg'] = data.close.rolling(window_size).mean()
    data['trend'] = 0

    for index in range(len(data)):
        moving_average_history = []
        if index >= window_size:
            for i in range(index - window_size, index):
                moving_average_history.append(data['sm_avg'][i])
        difference_moving_average = 0
        for i in range(len(moving_average_history) - 1, 0, -1):
            difference_moving_average += (moving_average_history[i] - moving_average_history[i - 1])

        # trend = 1 means ascending, and trend = 0 means descending
        data['trend'][index] = 1 if (difference_moving_average / window_size) > 0 else 0


def confirmation_of_the_trend(data, index):
    if index < len(data) - 1:
        return 'up' if data.close[index + 1] > data.close[index] else 'down'
