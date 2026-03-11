import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, hashlib, nltk
from itertools import combinations
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')
nltk.download('stopwords')

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
    print(df)
    return df

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f = 2 * p * r / (p + r) if p + r else 0
    return round(p, 3), round(r, 3), round(f, 3)

processed_docs = [" ".join(preprocess(doc)) for doc in docs]
shingles = [set(preprocess(doc)) for doc in docs]

# MinHash
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

# LSH
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

candidates = get_lsh_candidates(signature, 10)
lsh_df = pd.DataFrame([(f"Doc{i}", f"Doc{j}") for i, j in sorted(candidates)], columns=["Document 1", "Document 2"])
print("\nLSH Candidate Pairs Table")
print(lsh_df)

# Rocchio
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(processed_docs)
query = "information retrieval"
q_vec = vectorizer.transform([" ".join(preprocess(query))])
scores = cosine_similarity(q_vec, tfidf)[0]
top_docs = scores.argsort()[::-1][:3]
relevant = tfidf[top_docs]
non_relevant = tfidf[[i for i in range(len(docs)) if i not in top_docs]]

alpha, beta, gamma = 1, 0.75, 0.15
new_query = alpha * q_vec + beta * np.asarray(relevant.mean(axis=0)) - gamma * np.asarray(non_relevant.mean(axis=0))
new_scores = cosine_similarity(np.asarray(new_query), tfidf)[0]

rocchio_df = pd.DataFrame({
    "Document": [f"Doc{i}" for i in range(len(docs))],
    "Original Score": np.round(scores, 3),
    "Updated Score": np.round(new_scores, 3)
})
print("\nRocchio Score Table")
print(rocchio_df)

# LCA
top_k = scores.argsort()[::-1][:5]
term_freq = {}
for doc in [processed_docs[i] for i in top_k]:
    for word in doc.split():
        term_freq[word] = term_freq.get(word, 0) + 1
expanded_terms = sorted(term_freq, key=term_freq.get, reverse=True)[:5]
expanded_query = " ".join(preprocess(query)) + " " + " ".join(expanded_terms)
expanded_scores = cosine_similarity(vectorizer.transform([expanded_query]), tfidf)[0]

print("\nLCA Expanded Query")
print(expanded_query)

lca_df = pd.DataFrame({
    "Document": [f"Doc{i}" for i in range(len(docs))],
    "LCA Score": np.round(expanded_scores, 3)
})
print("\nLCA Score Table")
print(lca_df)

# Jaccard
jaccard = lambda a, b: len(a & b) / len(a | b)
jaccard_matrix = np.matrix([[jaccard(shingles[i], shingles[j]) for j in range(len(docs))] for i in range(len(docs))])
sim_df(jaccard_matrix, "Jaccard Similarity Table")

# Precision Recall Fscore with different bucket sizes
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
print("\nPrecision Recall Fscore with Different Bucket Size")
print(bucket_df)

# Signature Size Compression Ratio Accuracy
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
print("\nSignature Size Compression Ratio Accuracy Table")
print(compression_df)

# MAP change for different term reweighting
training_queries = ["information retrieval", "query expansion", "search engines", "duplicate detection"]
query_relevance = {
    "information retrieval": {0, 1, 2, 3, 4},
    "query expansion": {5, 6, 7},
    "search engines": {1, 3, 8},
    "duplicate detection": {8, 9}
}
settings = [(1.0, 0.75, 0.15), (1.0, 0.50, 0.25), (1.0, 1.00, 0.50)]

def avg_precision(score_vector, relevant_ids):
    ranked = np.argsort(score_vector)[::-1]
    hits, s = 0, 0
    for rank, d in enumerate(ranked, 1):
        if d in relevant_ids:
            hits += 1
            s += hits / rank
    return s / len(relevant_ids)

map_rows = []
for a, b, g in settings:
    before_list, after_list = [], []
    for tq in training_queries:
        tq_vec = vectorizer.transform([" ".join(preprocess(tq))])
        base = cosine_similarity(tq_vec, tfidf)[0]
        top = base.argsort()[::-1][:3]
        rel = tfidf[top]
        nonrel = tfidf[[i for i in range(len(docs)) if i not in top]]
        rq = a * tq_vec + b * np.asarray(rel.mean(axis=0)) - g * np.asarray(nonrel.mean(axis=0))
        updated = cosine_similarity(np.asarray(rq), tfidf)[0]
        before_list.append(avg_precision(base, query_relevance[tq]))
        after_list.append(avg_precision(updated, query_relevance[tq]))
    mb, ma = np.mean(before_list), np.mean(after_list)
    change = ((ma - mb) / mb) * 100 if mb else 0
    map_rows.append([a, b, g, round(mb, 3), round(ma, 3), round(change, 3)])

map_df = pd.DataFrame(map_rows, columns=["Alpha", "Beta", "Gamma", "MAP Before", "MAP After", "Percent Change in MAP"])
print("\nPercent Change in Mean Average Precision on Training Queries for Different Term Reweighting")
print(map_df)

# Graphs

# Additional Graphs

plt.figure()
plt.imshow(np.asarray(minhash_sim), cmap='viridis')
plt.colorbar()
plt.title("MinHash Similarity Heatmap")
plt.xlabel("Documents")
plt.ylabel("Documents")
plt.show()

plt.figure()
plt.imshow(np.asarray(jaccard_matrix), cmap='plasma')
plt.colorbar()
plt.title("Jaccard Similarity Heatmap")
plt.xlabel("Documents")
plt.ylabel("Documents")
plt.show()

plt.figure()
plt.bar(["Before Rocchio", "After Rocchio"], [np.mean(scores), np.mean(new_scores)])
plt.title("MAP Change After Rocchio")
plt.ylabel("MAP")
plt.show()

precision_val = bucket_df["Precision"].mean()
recall_val = bucket_df["Recall"].mean()
fscore_val = bucket_df["Fscore"].mean()

plt.figure()
plt.bar(["Precision", "Recall", "Fscore"], [precision_val, recall_val, fscore_val])
plt.title("Average Evaluation Metrics")
plt.ylabel("Value")
plt.show()

plt.figure()
plt.plot(bucket_df["Bucket Size"], bucket_df["Precision"], marker='o', label="Precision")
plt.plot(bucket_df["Bucket Size"], bucket_df["Recall"], marker='s', label="Recall")
plt.plot(bucket_df["Bucket Size"], bucket_df["Fscore"], marker='^', label="Fscore")
plt.title("PRF vs Bucket Size")
plt.xlabel("Bucket Size")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure()
plt.plot(compression_df["Signature Rows Used"], compression_df["Compression Ratio"], marker='o', label="Compression Ratio")
plt.plot(compression_df["Signature Rows Used"], compression_df["Accuracy"], marker='s', label="Accuracy")
plt.title("Compression Ratio and Accuracy")
plt.xlabel("Signature Rows Used")
plt.ylabel("Value")
plt.legend()
plt.show()

labels = [f"a={r['Alpha']}, b={r['Beta']}, g={r['Gamma']}" for _, r in map_df.iterrows()]

plt.figure()
plt.plot(labels, map_df["MAP Before"], marker='o', label="MAP Before")
plt.plot(labels, map_df["MAP After"], marker='s', label="MAP After")
plt.title("MAP for Different Reweighting")
plt.xlabel("Term Reweighting")
plt.ylabel("MAP")
plt.xticks(rotation=20)
plt.legend()
plt.show()

plt.figure()
plt.bar(labels, map_df["Percent Change in MAP"])
plt.title("Percent Change in MAP")
plt.xlabel("Term Reweighting")
plt.ylabel("Percent Change")
plt.xticks(rotation=20)
plt.show()