import numpy as np

from prepare_data import TrainValDataLoader, DataLoader, Query
from model import Model
from const import Const


def rmse(y_true: np.array, y_pred: np.array) -> np.float32:
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


if __name__ == '__main__':
    shape = (-1, Const.measurements_per_day, 1)

    train_observations: np.array
    train_forecasts: np.array
    train_observations, train_forecasts = TrainValDataLoader(
        [Query('Kyiv', '2012-07-01', '2013-07-01')]).get_data()
    train_forecasts = train_forecasts.reshape(shape)

    val_observations: np.array
    val_forecasts: np.array
    val_observations, val_forecasts = TrainValDataLoader(
        [Query('Kyiv', '2013-07-01', '2013-11-02')]).get_data()
    val_forecasts = val_forecasts.reshape(shape)

    model: Model = Model()
    model.train(train_forecasts, train_observations,
                (val_forecasts, val_observations))

    test_loader: DataLoader = DataLoader(
        Const.meteorological_stations['Teteriv'])
    test_observations: np.array
    test_forecasts: np.array
    test_observations, test_forecasts = test_loader.get_data()
    print("RMSE without ML:", rmse(test_forecasts, test_observations))

    test_forecasts = test_forecasts.reshape(shape)

    print("RMSE with ML (prediction):", rmse(
        model.forecast(test_forecasts), test_observations))
    print("RMSE with ML (evaluation):", np.sqrt(model.test(
        test_forecasts, test_observations)))
