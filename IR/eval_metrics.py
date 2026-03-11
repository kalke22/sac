import numpy as np
import pandas as pd
import random
import hashlib
import nltk
from itertools import combinations
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

random.seed(42)
np.random.seed(42)

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
    # print(df)
    return df

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f = 2 * p * r / (p + r) if p + r else 0
    return round(p, 3), round(r, 3), round(f, 3)

processed_docs = [" ".join(preprocess(doc)) for doc in docs]
shingles = [set(preprocess(doc)) for doc in docs]

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

# =====================================================================
# 1. JACCARD SIMILARITY
# =====================================================================
jaccard = lambda a, b: len(a & b) / len(a | b)
jaccard_matrix = np.matrix([[jaccard(shingles[i], shingles[j]) for j in range(len(docs))] for i in range(len(docs))])

sim_df(jaccard_matrix, "JACCARD SIMILARITY TABLE")
df_jaccard = pd.DataFrame(np.round(np.asarray(jaccard_matrix), 3))
print("=" * 60)
print(df_jaccard)

# =====================================================================
# 2. PRECISION, RECALL, FSCORE WITH DIFFERENT BUCKET SIZES (LSH)
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

threshold = 0.30
ground_truth = {(i, j) for i in range(len(docs)) for j in range(i + 1, len(docs)) if float(jaccard_matrix[i, j]) >= threshold}

bucket_rows = []
for b in [5, 10, 25]:
    if num_hash % b == 0:
        cand = get_lsh_candidates(signature, b)
        tp = len(cand & ground_truth)
        fp = len(cand - ground_truth)
        fn = len(ground_truth - cand)
        p, r, f = prf(tp, fp, fn)
        bucket_rows.append([b, len(cand), tp, fp, fn, p, r, f])

bucket_df = pd.DataFrame(bucket_rows, columns=["Bucket Size", "Candidate Pairs", "TP", "FP", "FN", "Precision", "Recall", "Fscore"])
print("\nPRECISION, RECALL, FSCORE WITH DIFFERENT BUCKET SIZES")
print("=" * 60)
print(bucket_df)

# =====================================================================
# 3. SIGNATURE SIZE / COMPRESSION RATIO AND ACCURACY
# =====================================================================
original_size = len(vocab) * len(docs)
comp_rows = []

for rows_used in [10, 20, 30, 40, 50]:
    sub_sig = signature[:rows_used, :]
    correct, total = 0, 0
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            approx = np.mean(sub_sig[:, i] == sub_sig[:, j]) >= threshold
            actual = float(jaccard_matrix[i, j]) >= threshold
            correct += int(approx == actual)
            total += 1
    comp_rows.append([
        rows_used,
        sub_sig.size,
        round(sub_sig.size / original_size, 3),
        round(correct / total, 3)
    ])

compression_df = pd.DataFrame(comp_rows, columns=["Signature Rows Used", "Signature Size", "Compression Ratio", "Accuracy"])
print("\nSIGNATURE SIZE, COMPRESSION RATIO & ACCURACY TABLE")
print("=" * 60)
print(compression_df)

# =====================================================================
# 4. PERCENT CHANGE IN MEAN AVERAGE PRECISION ON TRAINING QUERIES
# =====================================================================
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(processed_docs)

training_queries = ["information retrieval", "query expansion", "search engines", "duplicate detection"]

# Ground truth relevant documents mapping for training queries
query_relevance = {
    "information retrieval": {0, 1, 2, 3, 4},
    "query expansion": {5, 6, 7},
    "search engines": {1, 3, 8},
    "duplicate detection": {8, 9}
}

# (Alpha, Beta, Gamma) settings for Term Reweighting
settings = [(1.0, 0.75, 0.15), (1.0, 0.50, 0.25), (1.0, 1.00, 0.50)]

def avg_precision(score_vector, relevant_ids):
    ranked = np.argsort(score_vector)[::-1]
    hits, s = 0, 0
    for rank, d in enumerate(ranked, 1):
        if d in relevant_ids:
            hits += 1
            s += hits / rank
    return s / len(relevant_ids) if len(relevant_ids) > 0 else 0

map_rows = []
for a, b, g in settings:
    before_list, after_list = [], []
    for tq in training_queries:
        tq_vec = vectorizer.transform([" ".join(preprocess(tq))])
        base = cosine_similarity(tq_vec, tfidf)[0]
        
        # Pseudo-relevance for Rocchio on this query
        top = base.argsort()[::-1][:3]
        rel = tfidf[top]
        nonrel = tfidf[[i for i in range(len(docs)) if i not in top]]
        
        rq = a * tq_vec + b * np.asarray(rel.mean(axis=0)) - g * np.asarray(nonrel.mean(axis=0))
        updated = cosine_similarity(np.asarray(rq), tfidf)[0]
        
        # Calculate Average Precision
        before_list.append(avg_precision(base, query_relevance[tq]))
        after_list.append(avg_precision(updated, query_relevance[tq]))
        
    mb, ma = np.mean(before_list), np.mean(after_list)
    change = ((ma - mb) / mb) * 100 if mb else 0
    map_rows.append([a, b, g, round(mb, 3), round(ma, 3), round(change, 3)])

map_df = pd.DataFrame(map_rows, columns=["Alpha", "Beta", "Gamma", "MAP Before", "MAP After", "Percent Change in MAP"])
print("\nPERCENT CHANGE IN MEAN AVERAGE PRECISION ON TRAINING QUERIES")
print("=" * 60)
print(map_df)

# =====================================================================
# 5. VISUALIZATIONS
# =====================================================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(bucket_df["Bucket Size"], bucket_df["Precision"], marker='o', label="Precision")
plt.plot(bucket_df["Bucket Size"], bucket_df["Recall"], marker='s', label="Recall")
plt.plot(bucket_df["Bucket Size"], bucket_df["Fscore"], marker='^', label="Fscore")
plt.title("PRF vs Bucket Size")
plt.xlabel("Bucket Size")
plt.ylabel("Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(compression_df["Signature Rows Used"], compression_df["Compression Ratio"], marker='o', label="Compression")
plt.plot(compression_df["Signature Rows Used"], compression_df["Accuracy"], marker='s', label="Accuracy")
plt.title("Compression Ratio and Accuracy")
plt.xlabel("Signature Rows Used")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.savefig("eval_metrics_plots.png", dpi=150)
print("\nMetrics plots saved to 'eval_metrics_plots.png'.")
plt.show()
