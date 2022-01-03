import seaborn as sns
from matplotlib import pyplot as plt

from plots.services import save_plot

sns.set(style='white', palette='muted', color_codes=True, font_scale=1.2)


def plot_distribution(dataframe):
    class_names = ['negative', 'neutral', 'positive']
    ax = sns.countplot(dataframe)
    plt.xlabel('review sentiment')
    ax.set_xticklabels(class_names)
    save_plot('distribution')
