from matplotlib import pyplot as plt
from wordcloud import WordCloud

from plots.services import save_plot


def _get_corpus(dataframe):
    return dataframe.to_string(header=False, index=False)


def plot_word_cloud(text):
    wordcloud = WordCloud(width=480, height=480).generate(_get_corpus(text))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    save_plot('wordcloud')
