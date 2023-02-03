import numpy as np

from prepare_data import Query, TestDataLoader
from model import Model


def rmse(y_true: np.array, y_pred: np.array) -> np.float32:
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def get_forecast_nn(test_query: Query, model: Model, file_to_save: str) -> None:
    test_loader: TestDataLoader = TestDataLoader(test_query)

    test_observations: np.array
    test_forecasts: np.array
    test_observations, test_forecasts = test_loader.get_data()
    print("RMSE without ML:", rmse(test_forecasts, test_observations))

    test_forecasts_nn: np.array = model.forecast(test_forecasts)
    test_loader.update(test_forecasts_nn)
    test_loader.save_to_file(file_to_save)

    print("RMSE with ML (prediction):", rmse(
        test_forecasts_nn, test_observations))
    print("RMSE with ML (evaluation):", np.sqrt(model.test(
        test_forecasts, test_observations)))


if __name__ == '__main__':
    model: Model = Model()
    model.train([Query('Kyiv', '2012-07-01', '2013-07-01')],
                [Query('Kyiv', '2013-07-01', '2013-11-02')])

    get_forecast_nn(Query('Teteriv'), model, "results.csv")
