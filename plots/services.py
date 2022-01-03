from matplotlib import pyplot as plt

from common.config import PLOTS_PATH


def save_plot(figure_name: str) -> None:
    plt.savefig(f'{PLOTS_PATH}{figure_name}.png')
    plt.close()
