import numpy as np
import pandas as pd

from prepare_data import Query, TestDataLoader
from model import Model
from const import Const
from visualization_utils import draw_temperature_comparison


def rmse(y_true: np.array, y_pred: np.array) -> np.float32:
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def exp_00() -> None:
    loader: TestDataLoader = TestDataLoader(
        Query('Kyiv', '2013-04-01', '2013-05-01'))

    observations: np.array
    forecasts: np.array
    observations, forecasts = loader.get_data()
    observations = np.mean(observations, axis=0)
    forecasts = np.mean(forecasts, axis=0)

    categories: List[str] = [
        f'{num % 24}:00'.zfill(5) for num in range(3, 27, 3)]

    image_name: str = 'April 2013'
    draw_temperature_comparison(
        observations.tolist(), forecasts.tolist(), categories, image_name)


def exp_01() -> None:
    res = {}
    for st in Const.meteorological_stations:
        loader: TestDataLoader = TestDataLoader(Query(st))
        res[st] = f'{rmse(*loader.get_data()):.4f}'

    df = pd.DataFrame(data=res, index=['NWP']).transpose()
    df = df.reindex(sorted(df.index), axis=0)
    df.to_excel('../results/exp_01.xlsx', na_rep='NaN')


def exp_02() -> None:
    res = {}
    for st1 in Const.meteorological_stations:
        res[st1] = {}

        model: Model = Model()
        model.train([Query(st1, '2012-07-01', '2013-07-01')],
                    [Query(st1, '2013-07-01', '2013-11-02')])

        for st2 in Const.meteorological_stations:
            if st1 != st2:
                test_loader: TestDataLoader = TestDataLoader(Query(st2))

                test_observations: np.array
                test_forecasts: np.array
                test_observations, test_forecasts = test_loader.get_data()

                test_forecasts_nn: np.array = model.forecast(test_forecasts)
                test_loader.update(test_forecasts_nn)

                res[st1][st2] = f'{rmse(test_forecasts_nn, test_observations):.4f}'
            else:
                res[st1][st2] = 'NaN'

    df = pd.DataFrame(data=res)
    df = df.reindex(sorted(df.index), axis=0)
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_excel('../results/exp_02.xlsx', na_rep='NaN')


def exp_03():
    test_loader: TestDataLoader = TestDataLoader(
        Query('Kyiv', '2012-07-01', '2013-07-01'))

    test_observations: np.array
    test_observations, _ = test_loader.get_data()

    test_loader.update(test_observations)
    test_loader.save_to_file('results.csv')


if __name__ == '__main__':
    pass
