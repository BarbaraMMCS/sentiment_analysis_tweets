import spacy

from preprocessing.regex import NUMBERS_AND_SPECIAL_CHARACTERS_REGEX, SINGLE_CHARACTERS_REGEX, \
    SINGLE_CHARACTERS_FROM_START_REGEX, MULTIPLE_SPACE_REGEX

NLP = spacy.load("en_core_web_sm", disable=["ner"])

REGEX = [NUMBERS_AND_SPECIAL_CHARACTERS_REGEX, SINGLE_CHARACTERS_REGEX, SINGLE_CHARACTERS_FROM_START_REGEX,
         MULTIPLE_SPACE_REGEX]


def substitution_for_word_in(words, regex):
    return [regex.sub("", word) for word in words]


def preprocess_sentence(sentence):
    doc = NLP(sentence)
    words = [token.lemma_.lower() for token in doc]
    [substitution_for_word_in(words, regex) for regex in REGEX]
    return " ".join(words)


def add_preprocessing_sentences(play_store_review):
    play_store_review["processed_sentence"] = play_store_review.content.apply(preprocess_sentence)
