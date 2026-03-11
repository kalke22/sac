import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# =====================================================================
# READING FROM CORPUS (Example Code)
# =====================================================================
"""
To read documents from a local corpus directory:

import os
corpus_dir = "path/to/your/corpus/folder"
docs = []
for filename in os.listdir(corpus_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(corpus_dir, filename), "r", encoding="utf-8") as f:
            docs.append(f.read())
"""

docs = [
    "information retrieval is the process of obtaining relevant documents",
    "search engines use ranking algorithms for information retrieval",
    "information retrieval systems index and rank documents",
    "retrieval models help search engines find relevant documents",
    "inverted index is widely used in information retrieval",
    "query expansion improves retrieval effectiveness",
    "query expansion adds related terms to the query",
    "expansion techniques improve search results",
    "duplicate documents appear frequently in search engines",
    "near duplicate detection improves indexing"
]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    return [stemmer.stem(w) for w in word_tokenize(text.lower()) if w.isalnum() and w not in stop_words]

processed_docs = [" ".join(preprocess(doc)) for doc in docs]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(processed_docs)
query = "information retrieval"
processed_query = " ".join(preprocess(query))
q_vec = vectorizer.transform([processed_query])

# =====================================================================
# 1. ROCCHIO'S FEEDBACK ALGORITHM
# =====================================================================
scores = cosine_similarity(q_vec, tfidf)[0]

# Pseudo-relevance assumption: top 3 docs are relevant (in a real system, top 10-20)
num_pseudo_relevant = 3
top_docs = scores.argsort()[::-1][:num_pseudo_relevant]

alpha, beta, gamma = 1.0, 0.75, 0.15

relevant = tfidf[top_docs]
non_relevant = tfidf[[i for i in range(len(docs)) if i not in top_docs]]

new_query = alpha * q_vec + beta * np.asarray(relevant.mean(axis=0)) - gamma * np.asarray(non_relevant.mean(axis=0))
new_scores = cosine_similarity(np.asarray(new_query), tfidf)[0]

rocchio_df = pd.DataFrame({
    "Document": [f"Doc{i}" for i in range(len(docs))],
    "Original Score": np.round(scores, 3),
    "Updated Score (Rocchio)": np.round(new_scores, 3)
})
print("\nROCCHIO ALGORITHM SCORE TABLE")
print("=" * 60)
print(rocchio_df)

# =====================================================================
# 2. LOCAL CONTEXT ANALYSIS (LCA)
# =====================================================================
# Measures the co-occurrence of a term with all query terms based on information 
# from pseudo-relevant documents (top 10-20 documents returned by initial search).
# Since our corpus is small, we'll use top 5 pseudo-relevant documents.

num_lca_pseudo_relevant = 5
top_k_lca = scores.argsort()[::-1][:num_lca_pseudo_relevant]
top_docs_lca = [processed_docs[i] for i in top_k_lca]

term_freq = {}
for doc in top_docs_lca:
    for word in doc.split():
        term_freq[word] = term_freq.get(word, 0) + 1

# Extract top expanded terms from pseudo-relevant docs
num_expansion_terms = 5
expanded_terms = sorted(term_freq, key=term_freq.get, reverse=True)[:num_expansion_terms]
expanded_query = processed_query + " " + " ".join(expanded_terms)

expanded_vec = vectorizer.transform([expanded_query])
expanded_scores = cosine_similarity(expanded_vec, tfidf)[0]

print("\nLOCAL CONTEXT ANALYSIS (LCA)")
print("=" * 60)
print(f"Original Query: {processed_query}")
print(f"Expanded Query: {expanded_query}")

lca_df = pd.DataFrame({
    "Document": [f"Doc{i}" for i in range(len(docs))],
    "Original Score": np.round(scores, 3),
    "LCA Expanded Score": np.round(expanded_scores, 3)
})
print("\nLCA SCORE TABLE")
print(lca_df)
