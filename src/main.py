import numpy as np
import pandas as pd

from prepare_data import Query, TrainValDataLoader, TestDataLoader, CosmoSampler, ClimateSampler, Samples
from model import CosmoModel, Model, ClimateModel
from const import Const
from visualization_utils import draw_temperature_comparison


def rmse(y_true: np.array, y_pred: np.array) -> np.float32:
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def improvement_fraction(y_true: np.array, y_pred_nwp: np.array, y_pred_nn: np.array) -> np.float32:
    tmp: np.array = np.less(np.square(
        y_pred_nn - y_true), np.square(np.reshape(y_pred_nwp, y_true.shape) - y_true))
    return np.float32(tmp.sum()) / tmp.size


def mean_improvement(y_true: np.array, y_pred_nwp: np.array, y_pred_nn: np.array) -> None:
    tmp1 = np.abs(y_pred_nn - y_true).flatten()
    tmp2 = np.abs(np.reshape(y_pred_nwp, y_true.shape) - y_true).flatten()
    tmp = tmp2 - tmp1
    print(
        f'MEAN_IMPR: {np.mean(tmp[tmp < 0]):.4f}, {np.mean(tmp[tmp > 0]):.4f}')


def exp_00() -> None:
    loader: TestDataLoader = TestDataLoader(
        Query('Kyiv', '2013-04-01', '2013-05-01'), CosmoSampler())

    forecasts: np.array
    observations: np.array
    forecasts, observations = loader.get_data()
    forecasts = np.reshape(forecasts, observations.shape)
    forecasts = np.mean(forecasts, axis=0)
    observations = np.mean(observations, axis=0)

    categories: List[str] = [
        f'{num % 24}:00'.zfill(5) for num in range(3, 27, 3)]

    image_name: str = 'April 2013'
    draw_temperature_comparison(
        observations.tolist(), forecasts.tolist(), categories, image_name)


def exp_01() -> None:
    res = {}
    for st in Const.meteorological_stations:
        loader: TestDataLoader = TestDataLoader(Query(st), CosmoSampler())

        forecasts: np.array
        observations: np.array
        forecasts, observations = loader.get_data()
        forecasts = np.reshape(forecasts, observations.shape)

        res[st] = f'{rmse(forecasts, observations):.4f}'

    df = pd.DataFrame(data=res, index=['NWP']).transpose()
    df = df.reindex(sorted(df.index), axis=0)
    df.to_excel('../results/exp_01.xlsx', na_rep='NaN')


def exp_02() -> None:
    res = {}
    for st1 in Const.meteorological_stations:
        res[st1] = {}

        model: Model = CosmoModel()
        model.train([Query(st1, '2012-07-01', '2013-07-01')],
                    [Query(st1, '2013-07-01', '2013-11-02')], CosmoSampler())

        for st2 in Const.meteorological_stations:
            if st1 != st2:
                test_loader: TestDataLoader = TestDataLoader(
                    Query(st2), CosmoSampler())

                test_forecasts: np.array
                test_observations: np.array
                test_forecasts, test_observations = test_loader.get_data()

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
        Query('Kyiv', '2012-07-01', '2013-07-01'), CosmoSampler())

    test_observations: np.array
    _, test_observations = test_loader.get_data()

    test_loader.update(test_observations)
    test_loader.save_to_file('results_cosmo.csv')


def exp_04():
    loader: TestDataLoader = TestDataLoader(
        Query('Kyiv', '1965-07-01', '1965-08-01'), ClimateSampler())

    out: np.array
    _, out = loader.get_data()

    loader.update(out)
    loader.save_to_file('results_climate.csv')


def exp_05():
    samples: Samples = Samples()
    samples.add_arrays(
        np.array([[1, np.nan, np.nan, np.nan, 6, 7, 3]]), np.array([[2]]))
    samples.add_arrays(
        np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, 3]]), np.array([[2]]))
    samples.add_arrays(np.array([[1, 2, 3, 4, 5, 6, 7]]), np.array([[2]]))
    print(samples.get())
    samples = ClimateSampler().augment_samples(samples)
    print(samples.get())


def exp_06() -> None:
    Const.batch_size = 32
    Const.verbose = 2

    model: Model = ClimateModel()
    model.train([Query('Kyiv', '1961-01-01', '2000-12-31')],
                [Query('Kyiv', '2001-01-01', '2005-12-31')], ClimateSampler())

    for year in [2006, 2007, 2008, 2009, 2010]:
        loader: TestDataLoader = TestDataLoader(Query('Kyiv', str(np.datetime64(
            f'{year}-01-01') - np.timedelta64(Const.inp_vec_size + Const.forecast_len - 1, 'D')), f'{year}-12-31'), ClimateSampler())

        inp, out = loader.get_data()
        forecasts_nn: np.array = model.forecast(inp)
        loader.update(forecasts_nn)

        print(f'{rmse(forecasts_nn, out):.4f}')

        loader.save_to_file(f'results_climate_{year}.csv', Query(
            'Kyiv', f'{year}-01-01', f'{year}-12-31'))


def exp_07() -> None:
    train_queries = [Query(st, '2012-07-01', '2013-07-01')
                     for st in Const.meteorological_stations]
    val_queries = [Query(st, '2013-07-01', '2013-10-01')
                   for st in Const.meteorological_stations]

    model: Model = CosmoModel()
    model.train(train_queries, val_queries, CosmoSampler())

    test_queries = [Query(st, '2013-10-01', '2013-11-02')
                    for st in Const.meteorological_stations]

    test_forecasts: np.array
    test_observations: np.array
    test_forecasts, test_observations = TrainValDataLoader(
        test_queries, CosmoSampler()).get_data()

    test_forecasts_nn: np.array = model.forecast(test_forecasts)
    print(f'RMSE: {rmse(np.reshape(test_forecasts, test_observations.shape), test_observations):.4f} --> {rmse(test_forecasts_nn, test_observations):.4f}')
    print(
        f'FRAC: {improvement_fraction(test_observations, test_forecasts, test_forecasts_nn):.4f}')


def exp_08() -> None:
    train_queries_wo_kyiv = [Query(st, '2012-07-01', '2013-07-01')
                             for st in Const.meteorological_stations if st != 'Kyiv']
    val_queries_wo_kyiv = [Query(st, '2013-07-01', '2013-10-01')
                           for st in Const.meteorological_stations if st != 'Kyiv']

    model_wo_kyiv: Model = CosmoModel()
    model_wo_kyiv.train(train_queries_wo_kyiv,
                        val_queries_wo_kyiv, CosmoSampler())

    model_kyiv: Model = CosmoModel()
    model_kyiv.train([Query('Kyiv', '2012-07-01', '2013-07-01')],
                     [Query('Kyiv', '2013-07-01', '2013-10-01')], CosmoSampler())

    test_queries_wo_kyiv = [Query(st, '2013-10-01', '2013-11-02')
                            for st in Const.meteorological_stations if st != 'Kyiv']

    test_data_wo_kyiv = TrainValDataLoader(
        test_queries_wo_kyiv, CosmoSampler()).get_data()
    test_data_kyiv = TrainValDataLoader(
        [Query('Kyiv', '2013-10-01', '2013-11-02')], CosmoSampler()).get_data()

    for model in [model_wo_kyiv, model_kyiv]:
        for data in [test_data_wo_kyiv, test_data_kyiv]:
            test_forecasts: np.array
            test_observations: np.array
            test_forecasts, test_observations = data
            test_forecasts_nn: np.array = model.forecast(test_forecasts)
            print(f'RMSE: {rmse(np.reshape(test_forecasts, test_observations.shape), test_observations):.4f} --> {rmse(test_forecasts_nn, test_observations):.4f}')
            print(
                f'FRAC: {improvement_fraction(test_observations, test_forecasts, test_forecasts_nn):.4f}')
            mean_improvement(test_observations,
                             test_forecasts, test_forecasts_nn)


def exp_09() -> None:
    stations = [st for st in Const.meteorological_stations if st not in [
        'Teteriv', 'Chornobyl', 'Kyiv']]
    train_queries = [Query(st, '2012-07-01', '2013-07-01')
                     for st in stations]
    val_queries = [Query(st, '2013-07-01', '2013-10-01')
                   for st in stations]

    model: Model = CosmoModel()
    model.train(train_queries, val_queries, CosmoSampler())

    test_queries = [Query(st, '2013-10-01', '2013-11-02')
                    for st in Const.meteorological_stations]

    test_forecasts: np.array
    test_observations: np.array
    test_forecasts, test_observations = TrainValDataLoader(
        test_queries, CosmoSampler()).get_data()

    test_forecasts_nn: np.array = model.forecast(test_forecasts)
    print(f'RMSE: {rmse(np.reshape(test_forecasts, test_observations.shape), test_observations):.4f} --> {rmse(test_forecasts_nn, test_observations):.4f}')
    print(
        f'FRAC: {improvement_fraction(test_observations, test_forecasts, test_forecasts_nn):.4f}')


def exp_10(st1: str) -> None:
    model: Model = CosmoModel()
    model.train([Query(st1, '2012-07-01', '2013-07-01')],
                [Query(st1, '2013-07-01', '2013-11-02')], CosmoSampler())

    for st2 in Const.meteorological_stations:
        if st1 != st2:
            test_loader: TestDataLoader = TestDataLoader(
                Query(st2), CosmoSampler())

            test_forecasts: np.array
            test_observations: np.array
            test_forecasts, test_observations = test_loader.get_data()

            test_forecasts_nn: np.array = model.forecast(test_forecasts)
            print(st2)
            print(f'RMSE: {rmse(np.reshape(test_forecasts, test_observations.shape), test_observations):.4f} --> {rmse(test_forecasts_nn, test_observations):.4f}')
            print(
                f'FRAC: {improvement_fraction(test_observations, test_forecasts, test_forecasts_nn):.4f}')
            mean_improvement(test_observations,
                             test_forecasts, test_forecasts_nn)


if __name__ == '__main__':
    pass
