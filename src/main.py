from typing import List, Dict
import numpy as np

from prepare_data import DataLoader

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

if __name__ == '__main__':
    loader: DataLoader = DataLoader(meteorological_stations['Kyiv'])

    train_observations: np.array
    train_forecasts: np.array
    train_observations, train_forecasts = loader.get_data(
        '2012-07-01', '2013-07-01')

    val_observations: np.array
    val_forecasts: np.array
    val_observations, val_forecasts = loader.get_data(
        '2013-07-01', '2013-11-02')
