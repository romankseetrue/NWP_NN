from enum import Enum
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from const import Const
from dataclasses import dataclass


class SeriesType(Enum):
    OBSERVATION = 'Obs'
    FORECAST = 'Fcst'
    FORECAST_NN = 'FcstNN'


@dataclass
class Query:
    station_name: str = ''
    start_date: str = '2012-07-01'
    end_date: str = '2015-08-01'


class TrainValDataLoader:
    def __init__(self, queries: List[Query]) -> None:
        self.__queries: List[Query] = queries

    def get_data(self) -> Tuple[np.array, np.array]:
        res_observations: np.array = None
        res_forecasts: np.array = None
        for query in self.__queries:
            loader: DataLoader = DataLoader(query)
            observations, forecasts, _ = loader.get_data()
            res_observations = observations if res_observations is None else np.concatenate(
                (res_observations, observations), axis=0)
            res_forecasts = forecasts if res_forecasts is None else np.concatenate(
                (res_forecasts, forecasts), axis=0)
        return res_observations, res_forecasts


class DataLoader:
    def __init__(self, query: Query) -> None:
        self.__query = query
        self.__df: pd.DataFrame = process_file(
            Const.meteorological_stations[self.__query.station_name])

    def get_data(self) -> Tuple[np.array, np.array, List[int]]:
        range: pd.DatetimeIndex = pd.date_range(
            start=f'{self.__query.start_date} 03:00:00', end=f'{self.__query.end_date} 00:00:00', freq='3H')
        return treat_missing_values(self.__df[self.__df['DateTime'].isin(range)])


class TestDataLoader(DataLoader):
    def __init__(self, query: Query) -> None:
        super().__init__(query)
        self.__shape: Optional[Tuple] = None
        self.__indices_to_remove: Optional[List[int]] = None

    def get_data(self) -> Tuple[np.array, np.array]:
        observations, forecasts, self.__indices_to_remove = super().get_data()

        assert observations.shape == forecasts.shape
        self.__shape = observations.shape

        return observations, forecasts

    def update(self, forecasts_nn: np.array) -> None:
        assert forecasts_nn.shape == self.__shape
        assert self.__indices_to_remove

        for ind in self.__indices_to_remove:
            forecasts_nn = np.insert(forecasts_nn, ind, np.array(
                [np.nan] * forecasts_nn.shape[1]), axis=0)

        self._DataLoader__df[SeriesType.FORECAST_NN.value] = forecasts_nn.flatten(
        )

    def save_to_file(self, filepath: str) -> None:
        df = self._DataLoader__df.copy()
        df['DATE'] = df['DateTime'].apply(
            lambda ts: ts.date().strftime('%d.%m.%Y'))
        df['TIME'] = df['DateTime'].apply(
            lambda ts: ts.time().strftime('%H:%M'))
        df = df.drop(['DateTime'], axis=1).iloc[:, [3, 4, 0, 2, 1]]
        df.to_csv(filepath, na_rep='n', float_format='%.1f', index=False)


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
    series: np.array = np.array(df[type.value])

    assert series.size % Const.measurements_per_day == 0

    return np.reshape(series, (series.size // Const.measurements_per_day, Const.measurements_per_day))


def treat_missing_values(df: pd.DataFrame) -> Tuple[np.array, np.array, List[int]]:
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

    return observations, forecasts, indices_to_remove
