from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYSER = SentimentIntensityAnalyzer()

def _get_polarity_from_vader_score(vader_score):
    compound_score = vader_score.get("compound")
    if compound_score > 0.05:
        return "positive"
    elif compound_score < -0.05:
        return "negative"
    return "neutral"

def add_scores_and_polarities_to(play_store_review):
    play_store_review["vader_scores"] = play_store_review.content.apply(ANALYSER.polarity_scores)
    play_store_review["polarity_value"] = play_store_review.vader_scores.apply(_get_polarity_from_vader_score)
