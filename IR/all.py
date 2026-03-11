import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import hashlib
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')

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
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

processed_docs = [" ".join(preprocess(doc)) for doc in docs]

shingles = []
for doc in docs:
    tokens = preprocess(doc)
    shingles.append(set(tokens))

# MinHash

num_hash = 50
max_shingle = 1000
hash_funcs = []

for i in range(num_hash):
    a = random.randint(1, max_shingle)
    b = random.randint(0, max_shingle)
    hash_funcs.append((a, b))

def hash_function(x, a, b):
    return (a * x + b) % max_shingle

vocab = list(set(word for doc in shingles for word in doc))
shingle_index = {word: i for i, word in enumerate(vocab)}

signature = np.full((num_hash, len(docs)), np.inf)

for doc_id, doc in enumerate(shingles):
    for word in doc:
        idx = shingle_index[word]
        for i, (a, b) in enumerate(hash_funcs):
            h = hash_function(idx, a, b)
            signature[i, doc_id] = min(signature[i, doc_id], h)

signature = signature.astype(int)

def minhash_similarity(sig1, sig2):
    return np.mean(sig1 == sig2)

minhash_sim = np.matrix(np.zeros((len(docs), len(docs))))

for i in range(len(docs)):
    for j in range(len(docs)):
        minhash_sim[i, j] = minhash_similarity(signature[:, i], signature[:, j])

print("\nMinHash Similarity Matrix")
print(np.round(minhash_sim, 3))

# LSH

bands = 10
rows = int(num_hash / bands)
buckets = {}

for b in range(bands):
    for doc_id in range(len(docs)):
        band = tuple(signature[b * rows:(b + 1) * rows, doc_id])
        bucket_key = hashlib.md5(str(band).encode()).hexdigest()
        buckets.setdefault((b, bucket_key), []).append(doc_id)

candidates = set()

for bucket_docs in buckets.values():
    if len(bucket_docs) > 1:
        for pair in combinations(bucket_docs, 2):
            candidates.add(pair)

print("\nLSH Candidate Pairs")
print(candidates)

# Rocchio Algorithm

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(processed_docs)

query = "information retrieval"
processed_query = " ".join(preprocess(query))
q_vec = vectorizer.transform([processed_query])

scores = cosine_similarity(q_vec, tfidf)[0]
top_docs = scores.argsort()[::-1][:3]

alpha = 1
beta = 0.75
gamma = 0.15

relevant = tfidf[top_docs]
non_relevant = tfidf[[i for i in range(len(docs)) if i not in top_docs]]

new_query = alpha * q_vec + beta * np.asarray(relevant.mean(axis=0)) - gamma * np.asarray(non_relevant.mean(axis=0))
new_query = np.asarray(new_query)
new_scores = cosine_similarity(new_query, tfidf)[0]

print("\nRocchio Original Scores")
print(np.round(scores, 3))

print("\nRocchio Updated Scores")
print(np.round(new_scores, 3))

# Local Context Analysis

top_k = scores.argsort()[::-1][:5]
top_docs_lca = [processed_docs[i] for i in top_k]

term_freq = {}

for doc in top_docs_lca:
    for word in doc.split():
        term_freq[word] = term_freq.get(word, 0) + 1

expanded_terms = sorted(term_freq, key=term_freq.get, reverse=True)[:5]
expanded_query = processed_query + " " + " ".join(expanded_terms)

expanded_vec = vectorizer.transform([expanded_query])
expanded_scores = cosine_similarity(expanded_vec, tfidf)[0]

print("\nLCA Expanded Query")
print(expanded_query)

print("\nLCA Scores")
print(np.round(expanded_scores, 3))

# Jaccard Similarity

def jaccard(a, b):
    return len(a.intersection(b)) / len(a.union(b))

jaccard_matrix = np.matrix(np.zeros((len(shingles), len(shingles))))

for i in range(len(shingles)):
    for j in range(len(shingles)):
        jaccard_matrix[i, j] = jaccard(shingles[i], shingles[j])

print("\nJaccard Similarity Matrix")
print(np.round(jaccard_matrix, 3))

# Precision Recall Fscore

def precision(tp, fp):
    return tp / (tp + fp) if tp + fp else 0

def recall(tp, fn):
    return tp / (tp + fn) if tp + fn else 0

def fscore(p, r):
    return 2 * p * r / (p + r) if p + r else 0

tp, fp, fn = 8, 2, 3

p = precision(tp, fp)
r = recall(tp, fn)
f = fscore(p, r)

print("\nPrecision:", round(p, 3))
print("Recall:", round(r, 3))
print("Fscore:", round(f, 3))

# Signature Size Compression Ratio

original_size = len(vocab) * len(docs)
signature_size = signature.size
compression_ratio = signature_size / original_size

print("\nSignature Size:", signature_size)
print("Compression Ratio:", round(compression_ratio, 3))

# Mean Average Precision Change

map_before = np.mean(scores)
map_after = np.mean(new_scores)
percent_change = ((map_after - map_before) / map_before) * 100

print("\nMAP Before Rocchio:", round(map_before, 3))
print("MAP After Rocchio:", round(map_after, 3))
print("Percent Change in MAP:", round(percent_change, 3))

# Graphs
minhash_table = []

for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        minhash_table.append([
            f"Doc{i}",
            f"Doc{j}",
            round(minhash_sim[i,j],3)
        ])

minhash_df = pd.DataFrame(minhash_table,
                          columns=["Document 1","Document 2","MinHash Similarity"])

jaccard_table = []

for i in range(len(shingles)):
    for j in range(i+1, len(shingles)):
        jaccard_table.append([
            f"Doc{i}",
            f"Doc{j}",
            round(jaccard_matrix[i,j],3)
        ])

jaccard_df = pd.DataFrame(jaccard_table,
                          columns=["Document 1","Document 2","Jaccard Similarity"])

print("\nJaccard Similarity Table")
print(jaccard_df)
print("\nMinHash Similarity Table")
print(minhash_df)

plt.figure()
plt.bar(["Before Rocchio", "After Rocchio"], [map_before, map_after])
plt.title("MAP Change after Rocchio")
plt.show()

plt.figure()
plt.bar(["Precision", "Recall", "Fscore"], [p, r, f])
plt.title("Evaluation Metrics")
plt.show()