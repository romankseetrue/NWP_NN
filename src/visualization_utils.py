import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from typing import List


def draw_temperature_comparison(observations: List[float],
                                forecasts: List[float],
                                categories: List[str],
                                image_name: str) -> None:
    assert len(observations) == len(forecasts)
    assert len(observations) == len(categories)

    locations: List[int] = list(range(len(categories)))
    bar_width: float = 0.4

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()
    obs_rects: BarContainer = ax.bar(
        [loc - bar_width / 2 for loc in locations], observations, bar_width, label='Observations')
    fcst_rects: BarContainer = ax.bar(
        [loc + bar_width / 2 for loc in locations], forecasts, bar_width, label='Forecasts')

    ax.set_ylabel('Temperature')
    ax.set_xlabel('Time')
    ax.set_title(image_name)
    ax.set_xticks(locations, categories)
    ax.legend()

    padding: int = 3
    float_fmt: str = '%.2f'
    fontsize: int = 6
    ax.bar_label(obs_rects, padding=padding, fmt=float_fmt, fontsize=fontsize)
    ax.bar_label(fcst_rects, padding=padding, fmt=float_fmt, fontsize=fontsize)

    fig.tight_layout()

    dpi: int = 199
    plt.savefig(image_name, dpi=dpi)
