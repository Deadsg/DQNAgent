from sklearn.feature_extraction.text import CountVectorizer

corpus = ["This is a simple text.", "Text processing example.", "Natural Language Processing is interesting."]

# Create a bag-of-words model using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
