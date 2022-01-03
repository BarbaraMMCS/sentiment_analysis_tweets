from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common.services import print_service
from plots.distribution import plot_confusion_matrix


def train_and_test_svm(df_training):
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('clf', LinearSVC(max_iter=10000, C=3, random_state=1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(df_training.processed_sentence, df_training.polarity_value,
                                                        test_size=0.2, random_state=0)

    text_clf.fit(X_train, y_train)

    y_pred = text_clf.predict(X_test)

    print_service("Classificatino report", classification_report(y_test, y_pred))

    cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cf_matrix)

    return text_clf
