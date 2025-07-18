from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

processed_docs = [preprocess(doc) for doc in newsgroups_data.data]

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(processed_docs)

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(tfidf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(lda, tfidf_feature_names, no_top_words)

normalized_topic_distribution = normalize(lda.transform(tfidf))

doc_topic_df = pd.DataFrame(normalized_topic_distribution, columns=[f'Topic {i}' for i in range(lda.n_components)])

print(doc_topic_df.head())
