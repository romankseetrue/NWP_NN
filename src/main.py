from typing import Dict
import numpy as np

from prepare_data import DataLoader
from model import Model
from const import Const

meteorological_stations: Dict[str, int] = {
    'Teteriv': 33228,
    'Chornobyl': 33231,
    'Fastiv': 33339,
    'Kyiv': 33345,
    'Boryspil': 33347,
    'Yagotyn': 33356,
    'Bila Tserkva': 33464,
    'Myronivka': 33466
}


def mse(y_true: np.array, y_pred: np.array) -> np.float32:
    return np.mean(np.square(y_pred - y_true))


shape = (-1, Const.measurements_per_day, 1)

if __name__ == '__main__':
    loader: DataLoader = DataLoader(meteorological_stations['Kyiv'])

    train_observations: np.array
    train_forecasts: np.array
    train_observations, train_forecasts = loader.get_data(
        '2012-07-01', '2013-07-01')
    train_forecasts = train_forecasts.reshape(shape)

    val_observations: np.array
    val_forecasts: np.array
    val_observations, val_forecasts = loader.get_data(
        '2013-07-01', '2013-11-02')
    val_forecasts = val_forecasts.reshape(shape)

    model: Model = Model()
    model.train(train_forecasts, train_observations,
                (val_forecasts, val_observations))

    test_loader: DataLoader = DataLoader(meteorological_stations['Teteriv'])
    test_observations: np.array
    test_forecasts: np.array
    test_observations, test_forecasts = loader.get_data()
    print("RMSE without ML:", mse(test_forecasts, test_observations))

    test_forecasts = test_forecasts.reshape(shape)

    print("RMSE with ML (prediction):", mse(
        model.forecast(test_forecasts), test_observations))
    print("RMSE with ML (evaluation):", model.test(
        test_forecasts, test_observations))
