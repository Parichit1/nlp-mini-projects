 NLP Mini Projects with Python

This repository contains three small Natural Language Processing (NLP) projects written in Python.  
The projects cover **data collection, preprocessing, augmentation, and sentiment analysis** — core steps in many AI and ML pipelines.

---

## Projects

### 1. Web Scraping
- Scraped program information from a website using `requests` and `BeautifulSoup`.
- Exported results to CSV.
- Demonstrates challenges with dynamic JavaScript-driven content.

### 2. Text Preprocessing & Data Augmentation
- Preprocessed tweets: cleaned, lowercased, and removed noise.
- Used a pre-trained **Word2Vec** model to insert semantically similar words.
- Augmented dataset size by 2x.

### 3. Lexicon-Based Sentiment Analysis
- Classified tweets using **positive/negative word lexicons**.
- Calculated sentiment scores (% positive vs % negative words).
- Evaluated model → **Accuracy ~46%, F1 Score ~47%**.
- Output results to CSV.

---

## 🛠 Requirements
- Python 3.10+
- Install libraries with:
```bash
pip install pandas nltk gensim scikit-learn beautifulsoup4 selenium requests
