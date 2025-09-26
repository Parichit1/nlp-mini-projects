# exercise3_sentiment.py
# COMP262 - Assignment 1 - Exercise 3
# Student: Parichit Upadhayay

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load dataset (robust read in case of irregular commas)
print("📂 Loading dataset...")
df = pd.read_csv("Artifical_inteligence_data.csv", engine="python", on_bad_lines="skip")

# Step 2: Drop user column
if "user" in df.columns:
    df = df.drop(columns=["user"])

# Step 3: Clean tweets
def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)      # remove @mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text) # remove punctuation/numbers
    text = text.lower().strip()
    return text

df["clean_tweet"] = df["text"].apply(clean_tweet)

# Step 4: Add tweet length
df["tweet_len"] = df["clean_tweet"].apply(lambda x: len(x.split()))

# Step 5: Load lexicons (positive and negative word lists from Lab Week 3)
positive_words = pd.read_csv("positive-words.txt", 
                             header=None, names=["word"], comment=";", encoding="latin-1")
negative_words = pd.read_csv("negative-words.txt", 
                             header=None, names=["word"], comment=";", encoding="latin-1")

positive_set = set(positive_words["word"])
negative_set = set(negative_words["word"])

# Step 6: Calculate % positive and % negative
def get_sentiment_scores(text):
    tokens = word_tokenize(text)
    if len(tokens) == 0:
        return 0, 0
    pos_hits = sum(1 for w in tokens if w in positive_set)
    neg_hits = sum(1 for w in tokens if w in negative_set)
    pos_percent = pos_hits / len(tokens)
    neg_percent = neg_hits / len(tokens)
    return pos_percent, neg_percent

df[["pos_percent", "neg_percent"]] = df["clean_tweet"].apply(
    lambda x: pd.Series(get_sentiment_scores(x))
)

# Step 7: Predict sentiment
def predict_sentiment(row):
    if row["pos_percent"] == row["neg_percent"]:
        return "neutral"
    elif row["pos_percent"] > row["neg_percent"]:
        return "positive"
    else:
        return "negative"

df["predicted_sentiment_score"] = df.apply(predict_sentiment, axis=1)

# Step 8: Evaluate
accuracy = accuracy_score(df["sentiment"], df["predicted_sentiment_score"])
f1 = f1_score(df["sentiment"], df["predicted_sentiment_score"], average="weighted")

print("\nSentiment analysis complete!")
print("Accuracy:", round(accuracy, 3))
print("F1 Score:", round(f1, 3))

# Save results
df.to_csv("parichit_sentiment_analysis.csv", index=False)
print("\n💾 Results saved as parichit_sentiment_analysis.csv")
