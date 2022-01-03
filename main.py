import pandas as pd

from common.config import REVIEWS_PATH, DATA_PATH
from dataframe_exploration.exploration import explore_dataframe
from ml.svm import train_and_test_svm
from plots.distribution import plot_distribution
from plots.wordcloud import plot_word_cloud
from preprocessing.preprocessing import add_preprocessing_sentences
from sentiment_analysis.vader import add_scores_and_polarities_to


def main():
    play_store_review_df = pd.read_csv(REVIEWS_PATH)

    explore_dataframe(play_store_review_df)
    add_scores_and_polarities_to(play_store_review_df)
    add_preprocessing_sentences(play_store_review_df)

    dataframe = play_store_review_df[["processed_sentence", "vader_scores", "polarity_value"]]
    dataframe.to_csv(DATA_PATH)
    explore_dataframe(dataframe)

    plot_word_cloud(dataframe['processed_sentence'])
    plot_distribution(dataframe)

    clf = train_and_test_svm(dataframe)


if __name__ == '__main__':
    main()

