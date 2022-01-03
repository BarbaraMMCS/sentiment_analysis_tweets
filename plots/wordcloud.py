from matplotlib import pyplot as plt
from wordcloud import WordCloud

from plots.services import save_plot


def plot_word_cloud(sentence):
    wordcloud = WordCloud(width=480, height=480).generate(" ".join())
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    save_plot('wordcloud')
