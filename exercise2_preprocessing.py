# exercise2_preprocessing.py
# COMP262 - Assignment 1 - Exercise 2
# Student: Parichit Upadhayay

import pandas as pd
import re
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader as api3

# Step 1: Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("Artificial_Intelligence_mini.csv")
parichit_df = df.copy()

# Step 2: Drop user column
if "user" in parichit_df.columns:
    parichit_df = parichit_df.drop(columns=["user"])

# Step 3: Clean tweets
def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)      # remove @mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text) # remove punctuation/numbers
    text = text.lower().strip()           # lowercase and strip spaces
    return text

parichit_df["clean_tweet"] = parichit_df["text"].apply(clean_tweet)

print("Cleaned Tweets Example:")
print(parichit_df[["text", "clean_tweet"]].head())

# Step 4: Load word2vec model
print("\n⏳ Loading Word2Vec model (this may take a while first time)...")
word2vec = api.load("word2vec-google-news-300")
print(" Word2Vec loaded!")

stop_words = set(stopwords.words("english"))

# Step 5: Augmentation function
def augment_tweet(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]

    if len(tokens) < 3:
        return text  

    new_tokens = tokens.copy()
    words_to_replace = random.sample(tokens, min(3, len(tokens)))

    for w in words_to_replace:
        try:
            synonyms = word2vec.most_similar(w, topn=5)
            replacement = random.choice(synonyms)[0]
            new_tokens[new_tokens.index(w)] = replacement
        except KeyError:
            continue  

    return " ".join(new_tokens)

# Step 6: Apply augmentation
print("\n🔄 Augmenting tweets...")
augmented_rows = []
for idx, row in parichit_df.iterrows():
    new_tweet = augment_tweet(row["clean_tweet"])
    augmented_rows.append({"text": new_tweet, "sentiment": row["sentiment"]})

augmented_df = pd.DataFrame(augmented_rows)

# Step 7: Merge old + new dataset
final_df = pd.concat([parichit_df[["clean_tweet", "sentiment"]], augmented_df], ignore_index=True)

# Step 8: Save new dataset
final_df.to_csv("parichit_df_after_random_insertion.csv", index=False)

print("\nAugmented dataset saved as parichit_df_after_random_insertion.csv")
print("Final dataset size:", final_df.shape)
print("\nExample Original vs Augmented Tweet:")
print("Original:", parichit_df['clean_tweet'].iloc[0])
print("Augmented:", augmented_df['text'].iloc[0])