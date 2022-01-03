import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from plots.services import save_plot

sns.set(style='white', palette='muted', color_codes=True, font_scale=1.2)
CLASS_NAME = ['neutral', 'positive', 'negative']


def plot_distribution(dataframe):
    ax = sns.countplot(dataframe['polarity_value'])
    plt.xlabel('review sentiment')
    ax.set_xticklabels(CLASS_NAME)
    save_plot('distribution')


def plot_confusion_matrix(cf_matrix):
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0: .2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    save_plot('confusion_matrix')
