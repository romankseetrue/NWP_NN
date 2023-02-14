from __future__ import annotations

from enum import Enum
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from const import Const
from dataclasses import dataclass


@dataclass
class Query:
    station_name: str = ''
    start_date: str = '2012-07-01'
    end_date: str = '2015-08-01'


class Samples:
    def __init__(self) -> None:
        self.__inp: Optional[np.array] = None
        self.__out: Optional[np.array] = None
        self.__num: int = 0

    def add_arrays(self, inp: np.array, out: np.array) -> None:
        self.__inp = inp if self.__inp is None else np.concatenate(
            (self.__inp, inp), axis=0)
        self.__out = out if self.__out is None else np.concatenate(
            (self.__out, out), axis=0)

        assert self.__inp.shape[0] == self.__out.shape[0]
        self.__num = self.__inp.shape[0]

    def add_samples(self, samples: Samples) -> None:
        self.add_arrays(samples.__inp, samples.__out)

    def get(self) -> Tuple[np.array, np.array]:
        return self.__inp, self.__out

    def size(self) -> int:
        return self.__num

    def remove_by_indices(self, indices_to_remove: List[int]) -> None:
        self.__inp = np.delete(self.__inp, indices_to_remove, axis=0)
        self.__out = np.delete(self.__out, indices_to_remove, axis=0)


class TrainValDataLoader:
    def __init__(self, queries: List[Query], samples_designer: Sampler) -> None:
        self.__queries: List[Query] = queries
        self.__samples_designer: Sampler = samples_designer

    def get_data(self) -> Tuple[np.array, np.array]:
        samples: Samples = Samples()
        for query in self.__queries:
            loader: DataLoader = DataLoader(query, self.__samples_designer)
            samples.add_arrays(*loader.get_data())
        return samples.get()


class Sampler:
    def get_samples(self) -> Samples:
        pass


class CosmoSampler(Sampler):
    def process_file(self, query: Query) -> pd.DataFrame:
        file_id: int = Const.meteorological_stations[query.station_name]
        file_name: str = f'../data/COSMO/fo_ua_all_cosmo_{file_id}_TTT_20120701_20131101_start00_zv03_trv21.csv'
        df: pd.DataFrame = pd.read_csv(
            file_name, skipinitialspace=True).replace('n', np.nan)

        df['DateTime'] = pd.to_datetime(
            df['DATE'] + ' ' + df['TIME'], dayfirst=True)
        df = df.drop(['DATE', 'TIME'], axis=1).iloc[:, [2, 0, 1]]

        df['Fcst'] = df['Fcst'].astype(float)
        df['Obs'] = df['Obs'].astype(float)

        range: pd.DatetimeIndex = pd.date_range(
            start=f'{query.start_date} 03:00:00', end=f'{query.end_date} 00:00:00', freq='3H')

        return df[df['DateTime'].isin(range)]

    def get_series(self, df: pd.DataFrame, type: str) -> np.array:
        series: np.array = np.array(df[type])

        assert series.size % Const.measurements_per_day == 0

        return np.reshape(series, (series.size // Const.measurements_per_day, Const.measurements_per_day))

    def get_samples(self, df: pd.DataFrame) -> Samples:
        samples: Samples = Samples()
        samples.add_arrays(self.get_series(df, 'Fcst'), self.get_series(
            df, 'Obs'))
        return samples

    def treat_missing_values(self, samples: Samples) -> List[int]:
        size: int = samples.size()

        indices_to_remove: List[int] = []
        for i in range(size):
            if np.any(np.isnan(samples.get()[0][i])) or np.any(np.isnan(samples.get()[1][i])):
                indices_to_remove.append(i)

        return indices_to_remove

    def save_to_file(self, df: pd.DataFrame, filepath: str) -> None:
        df['DATE'] = df['DateTime'].apply(
            lambda ts: ts.date().strftime('%d.%m.%Y'))
        df['TIME'] = df['DateTime'].apply(
            lambda ts: ts.time().strftime('%H:%M'))
        df = df.drop(['DateTime'], axis=1).iloc[:, [3, 4, 0, 2, 1]]
        df.to_csv(filepath, na_rep='n', float_format='%.1f', index=False)


class ClimateSampler(Sampler):
    inp_vec_size: int = 10
    forecast_len: int = 3

    def process_file(self, query: Query) -> pd.DataFrame:
        file_name: str = f'../data/Kyiv_Daily_Temperature_bl_1961-2010.csv'
        df: pd.DataFrame = pd.read_csv(
            file_name, skipinitialspace=True).replace('n', np.nan)

        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
        df['TG'] = df['TG'].astype(float)

        range: pd.DatetimeIndex = pd.date_range(
            start=f'{query.start_date}', end=f'{query.end_date}', freq='D')

        return df[df['DATE'].isin(range)]

    def get_samples(self, df: pd.DataFrame) -> Samples:
        series: np.array = np.array(df['TG'])

        samples: Samples = Samples()
        for ind in range(series.size):
            start_ind = ind - ClimateSampler.inp_vec_size - ClimateSampler.forecast_len + 1
            end_ind = ind - ClimateSampler.forecast_len + 1
            if start_ind < 0:
                samples.add_arrays(
                    np.full((1, ClimateSampler.inp_vec_size), np.nan), np.full((1, 1), np.nan))
            else:
                samples.add_arrays(np.reshape(
                    series[start_ind:end_ind], (1, ClimateSampler.inp_vec_size)), np.full((1, 1), series[ind]))
        return samples

    def treat_missing_values(self, samples: Samples) -> List[int]:
        size: int = samples.size()

        inp, out = samples.get()

        indices_to_remove: List[int] = []
        for i in range(size):
            if np.any(np.isnan(out[i])) or np.isnan(inp[i][0]) or np.isnan(inp[i][ClimateSampler.inp_vec_size - 1]):
                indices_to_remove.append(i)

        return indices_to_remove

    def save_to_file(self, df: pd.DataFrame, filepath: str) -> None:
        df['DATE'] = df['DATE'].apply(
            lambda ts: ts.date().strftime('%Y%m%d'))
        df.to_csv(filepath, na_rep='n', float_format='%.1f', index=False)


class DataLoader:
    def __init__(self, query: Query, samples_designer: Sampler) -> None:
        self.__indices_to_remove: Optional[List[int]] = None
        self.__samples_designer: Sampler = samples_designer
        self.__df: pd.DataFrame = self.__samples_designer.process_file(query)

    def get_data(self) -> Samples:
        samples: Samples = self.__samples_designer.get_samples(self.__df)
        self.__indices_to_remove = self.__samples_designer.treat_missing_values(
            samples)
        samples.remove_by_indices(self.__indices_to_remove)

        return samples.get()


class TestDataLoader(DataLoader):
    def update(self, forecasts_nn: np.array) -> None:
        assert self._DataLoader__indices_to_remove

        for ind in self._DataLoader__indices_to_remove:
            forecasts_nn = np.insert(forecasts_nn, ind, np.full(
                forecasts_nn.shape[1:], np.nan), axis=0)

        self._DataLoader__df['FcstNN'] = forecasts_nn.flatten()

    def save_to_file(self, filepath: str) -> None:
        df = self._DataLoader__df.copy()
        self._DataLoader__samples_designer.save_to_file(df, filepath)
