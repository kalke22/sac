import numpy as np
import pandas as pd
import random
import hashlib
import nltk
from itertools import combinations
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

random.seed(42)
np.random.seed(42)

# =====================================================================
# READING FROM CORPUS (Example Code)
# =====================================================================
"""
To read documents from a local corpus directory instead of the hardcoded list below,
you can use the following snippet:

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

def sim_df(mat, title):
    df = pd.DataFrame(np.round(np.asarray(mat), 3),
                      index=[f"Doc{i}" for i in range(len(docs))],
                      columns=[f"Doc{i}" for i in range(len(docs))])
    print(f"\n{title}")
    print(df)
    return df

processed_docs = [" ".join(preprocess(doc)) for doc in docs]
shingles = [set(preprocess(doc)) for doc in docs]

# =====================================================================
# 1. MINHASH
# =====================================================================
num_hash, max_shingle = 50, 1000
hash_funcs = [(random.randint(1, max_shingle), random.randint(0, max_shingle)) for _ in range(num_hash)]
vocab = list(set(word for doc in shingles for word in doc))
shingle_index = {w: i for i, w in enumerate(vocab)}

def h(x, a, b): return (a * x + b) % max_shingle

signature = np.full((num_hash, len(docs)), np.inf)
for d, doc in enumerate(shingles):
    for word in doc:
        idx = shingle_index[word]
        for i, (a, b) in enumerate(hash_funcs):
            signature[i, d] = min(signature[i, d], h(idx, a, b))
signature = signature.astype(int)

minhash_sim = np.matrix([[np.mean(signature[:, i] == signature[:, j]) for j in range(len(docs))] for i in range(len(docs))])
sim_df(minhash_sim, "MinHash Similarity Table")

# =====================================================================
# 2. LOCALITY SENSITIVE HASHING (LSH)
# =====================================================================
def get_lsh_candidates(sig, bands):
    rows = sig.shape[0] // bands
    buckets, candidates = {}, set()
    for b in range(bands):
        for d in range(sig.shape[1]):
            band = tuple(sig[b * rows:(b + 1) * rows, d])
            key = hashlib.md5(str(band).encode()).hexdigest()
            buckets.setdefault((b, key), []).append(d)
    for group in buckets.values():
        if len(group) > 1:
            for pair in combinations(group, 2):
                candidates.add(tuple(sorted(pair)))
    return candidates

bands = 10
candidates = get_lsh_candidates(signature, bands)
lsh_df = pd.DataFrame([(f"Doc{i}", f"Doc{j}") for i, j in sorted(candidates)], columns=["Document 1", "Document 2"])
print(f"\nLSH Candidate Pairs Table (Bands={bands})")
print(lsh_df)
