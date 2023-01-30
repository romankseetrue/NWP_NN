from typing import List, Dict
import numpy as np

from prepare_data import DataLoader

from visualization_utils import draw_temperature_comparison

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
