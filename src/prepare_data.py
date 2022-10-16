from enum import Enum
from typing import List, Tuple
import pandas as pd
import numpy as np


class SeriesType(Enum):
    OBSERVATION = 'Obs'
    FORECAST = 'Fcst'


class DataLoader:
    def __init__(self, file_id: int) -> None:
        self.__df: pd.DataFrame = process_file(file_id)

    def get_data(self, start_date: str = '2012-07-01',
                 end_date: str = '2015-08-01') -> Tuple[np.array, np.array]:
        range: pd.DatetimeIndex = pd.date_range(
            start=f'{start_date} 03:00:00', end=f'{end_date} 00:00:00', freq='3H')
        return treat_missing_values(self.__df[self.__df['DateTime'].isin(range)])


def process_file(file_id: int) -> pd.DataFrame:
    file_name: str = f'../data/fo_ua_all_cosmo_{file_id}_TTT_20120701_20131101_start00_zv03_trv21.csv'
    df: pd.DataFrame = pd.read_csv(
        file_name, skipinitialspace=True).replace('n', np.nan)

    df['DateTime'] = pd.to_datetime(
        df['DATE'] + ' ' + df['TIME'], dayfirst=True)
    df = df.drop(['DATE', 'TIME'], axis=1).iloc[:, [2, 0, 1]]

    df['Fcst'] = df['Fcst'].astype(float)
    df['Obs'] = df['Obs'].astype(float)

    return df


def get_series(df: pd.DataFrame, type: SeriesType) -> np.array:
    measurements_per_day: int = 8
    series: np.array = np.array(df[type.value])

    assert series.size % measurements_per_day == 0

    return np.reshape(series, (series.size // measurements_per_day, measurements_per_day))


def treat_missing_values(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    observations: np.array = get_series(df, SeriesType.OBSERVATION)
    forecasts: np.array = get_series(df, SeriesType.FORECAST)

    assert observations.shape[0] == forecasts.shape[0]

    size: int = observations.shape[0]

    indices_to_remove: List[int] = []
    for i in range(size):
        if np.any(np.isnan(observations[i])) or np.any(np.isnan(forecasts[i])):
            indices_to_remove.append(i)

    observations = np.delete(observations, indices_to_remove, axis=0)
    forecasts = np.delete(forecasts, indices_to_remove, axis=0)

    return observations, forecasts
