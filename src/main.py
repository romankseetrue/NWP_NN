from typing import List
import numpy as np

from prepare_data import DataLoader

from visualization_utils import draw_temperature_comparison


if __name__ == '__main__':
    loader: DataLoader = DataLoader(33345)
    start_date: str = '2012-07-01'
    end_date: str = '2012-08-01'
    image_name: str = 'July 2012'

    observations: np.array
    forecasts: np.array
    observations, forecasts = loader.get_data(start_date, end_date)
    observations = np.mean(observations, axis=0)
    forecasts = np.mean(forecasts, axis=0)

    categories: List[str] = [
        f'{num % 24}:00'.zfill(5) for num in range(3, 27, 3)]

    draw_temperature_comparison(
        observations.tolist(), forecasts.tolist(), categories, image_name)
