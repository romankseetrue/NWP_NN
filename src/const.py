from typing import Dict


class Const:
    measurements_per_day: int = 8
    batch_size: int = 1
    epochs: int = 50
    verbose: int = 0
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
