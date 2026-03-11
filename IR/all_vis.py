import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, hashlib, nltk, glob, os
from itertools import combinations
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt_tab')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def setup_seeds():
    random.seed(42)
    np.random.seed(42)

def preprocess(text):
    return [stemmer.stem(w) for w in word_tokenize(text.lower()) if w.isalnum() and w not in stop_words]

def get_k_shingles(words, k=2):
    if len(words) < k:
        return words
    return [" ".join(words[i:i+k]) for i in range(len(words) - k + 1)]

def load_corpus(directory="corpus"):
    docs = []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, directory)
    
    for file in glob.glob(os.path.join(target_dir, "*.txt")):
        with open(file, 'r', encoding='utf-8') as f: docs.append(f.read().strip())
            
    for file in glob.glob(os.path.join(target_dir, "*.csv")):
        docs.extend([str(item).strip() for row in pd.read_csv(file, header=None).values for item in row if isinstance(item, str)])
                    
    return docs

def print_table(df, title):
    width = max(len(title) + 4, 60)
    print(f"\n{'=' * width}\n {title} \n{'=' * width}\n\n{df}\n\n{'-' * width}\n")

def sim_df(mat, docs, title):
    df = pd.DataFrame(np.round(np.asarray(mat), 3),
                      index=[f"Doc{i}" for i in range(len(docs))],
                      columns=[f"Doc{i}" for i in range(len(docs))])
    print_table(df, title)
    return df

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f = 2 * p * r / (p + r) if p + r else 0
    return round(p, 3), round(r, 3), round(f, 3)

def compute_minhash(shingles, docs, num_hash=50, max_shingle=1000):
    hash_funcs = [(random.randint(1, max_shingle), random.randint(0, max_shingle)) for _ in range(num_hash)]
    vocab = list(set(word for doc in shingles for word in doc)) or ["empty"]
    shingle_index = {w: i for i, w in enumerate(vocab)}

    signature = np.full((num_hash, len(docs)), np.inf)
    for d, doc in enumerate(shingles):
        for word in doc:
            for i, (a, b) in enumerate(hash_funcs):
                signature[i, d] = min(signature[i, d], (a * shingle_index[word] + b) % max_shingle)
    signature = signature.astype(int)

    minhash_sim = np.matrix([[np.mean(signature[:, i] == signature[:, j]) for j in range(len(docs))] for i in range(len(docs))])
    sim_df(minhash_sim, docs, "MinHash Similarity Table")
    return signature, minhash_sim, vocab, num_hash

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

def compute_lsh(signature, bands=50):
    candidates = get_lsh_candidates(signature, bands)
    lsh_df = pd.DataFrame([(f"Doc{i}", f"Doc{j}") for i, j in sorted(candidates)], columns=["Document 1", "Document 2"])
    
    print_table(lsh_df, "LSH Candidate Pairs Table (Near Duplicates)")
    return candidates
    
    return candidates

def compute_rocchio(processed_docs, docs, query="information retrieval"):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(processed_docs)
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
    
    print_table(rocchio_df, "Rocchio Score Table")
    return vectorizer, tfidf, scores, new_scores

def compute_lca(processed_docs, docs, vectorizer, tfidf, scores, query="information retrieval"):
    top_k = scores.argsort()[::-1][:5]
    expanded_terms = [w for w, _ in Counter(word for i in top_k for word in processed_docs[i].split()).most_common(5)]
    expanded_query = " ".join(preprocess(query)) + " " + " ".join(expanded_terms)
    expanded_scores = cosine_similarity(vectorizer.transform([expanded_query]), tfidf)[0]

    width = 60
    print(f"\n{'=' * width}\n Query Expansion Comparison \n{'=' * width}\n")
    print(f"Before:  {query}")
    print(f"After :  {expanded_query}\n")
    print(f"{'-' * width}\n")

    lca_df = pd.DataFrame({
        "Document": [f"Doc{i}" for i in range(len(docs))],
        "LCA Score": np.round(expanded_scores, 3)
    })
    
    print_table(lca_df, "LCA Score Table")
    return expanded_scores

def compute_jaccard(shingles, docs):
    jaccard = lambda a, b: len(a & b) / len(a | b) if len(a | b) > 0 else 0
    jaccard_matrix = np.matrix([[jaccard(shingles[i], shingles[j]) for j in range(len(docs))] for i in range(len(docs))])
    sim_df(jaccard_matrix, docs, "Jaccard Similarity Table")
    return jaccard_matrix

def evaluate_prf(signature, jaccard_matrix, docs, num_hash):
    threshold = 0.10 
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
    
    print_table(bucket_df, "Precision Recall Fscore with Different Bucket Size")
    return bucket_df

def evaluate_compression(signature, jaccard_matrix, docs, vocab):
    threshold = 0.10
    original_size = len(vocab) * len(docs)
    total_pairs = len(docs) * (len(docs) - 1) // 2
    comp_rows = []
    
    for rows_used in [10, 20, 30, 40, 50]:
        sub_sig = signature[:rows_used, :]
        correct = sum((np.mean(sub_sig[:, i] == sub_sig[:, j]) >= threshold) == (float(jaccard_matrix[i, j]) >= threshold) 
                      for i, j in combinations(range(len(docs)), 2))
        
        comp_rows.append([
            rows_used, sub_sig.size,
            round(sub_sig.size / original_size, 3) if original_size else 0,
            round(correct / total_pairs, 3) if total_pairs else 0
        ])

    compression_df = pd.DataFrame(comp_rows, columns=["Signature Rows Used", "Signature Size", "Compression Ratio", "Accuracy"])
    
    print_table(compression_df, "Signature Size Compression Ratio Accuracy Table")
    return compression_df

def avg_precision(score_vector, relevant_ids):
    ranked = np.argsort(score_vector)[::-1]
    hits, s = 0, 0
    for rank, d in enumerate(ranked, 1):
        if d in relevant_ids:
            hits += 1
            s += hits / rank
    return s / len(relevant_ids) if relevant_ids else 0

def evaluate_map(vectorizer, tfidf, docs):
    training_queries = ["information retrieval", "query expansion", "search engines", "duplicate detection"]
    query_relevance = {
        "information retrieval": {0, 1, 2, 3, 4},
        "query expansion": {5, 6, 7},
        "search engines": {1, 3, 8},
        "duplicate detection": {8, 9}
    }
    settings = [(1.0, 0.75, 0.15), (1.0, 0.50, 0.25), (1.0, 1.00, 0.50)]

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
    
    print_table(map_df, "Percent Change in Mean Average Precision on Training Queries for Different Term Reweighting")
    return map_df

def plot_graphs(minhash_sim, jaccard_matrix, scores, new_scores, bucket_df, compression_df, map_df, docs, shingles):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Corpus Properties")
    
    doc_lengths = [len(s) for s in shingles]
    axes[0].bar([f"D{i}" for i in range(len(docs))], doc_lengths, color='skyblue')
    axes[0].set_title("Number of K-Shingles per Document")
    axes[0].set_xlabel("Documents")
    axes[0].set_ylabel("Shingle Count")
    
    x = np.arange(len(docs))
    width = 0.35
    axes[1].bar(x - width/2, scores, width, label='Original VSM Score')
    axes[1].bar(x + width/2, new_scores, width, label='Rocchio Score')
    axes[1].set_title("Rocchio Relevance Feedback Impact")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"D{i}" for i in range(len(docs))])
    axes[1].set_ylabel("Cosine Similarity Score")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Document Similarities")
    
    im1 = axes[0].imshow(np.asarray(minhash_sim), cmap='viridis')
    axes[0].set_title("MinHash Similarity Heatmap")
    axes[0].set_xlabel("Documents")
    axes[0].set_ylabel("Documents")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(np.asarray(jaccard_matrix), cmap='plasma')
    axes[1].set_title("Jaccard Similarity Heatmap")
    axes[1].set_xlabel("Documents")
    axes[1].set_ylabel("Documents")
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSH Evaluation Metrics")
    
    precision_val = bucket_df["Precision"].mean()
    recall_val = bucket_df["Recall"].mean()
    fscore_val = bucket_df["Fscore"].mean()
    
    axes[0].bar(["Precision", "Recall", "Fscore"], [precision_val, recall_val, fscore_val], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title("Average Classification Metrics")
    axes[0].set_ylabel("Mean Value")
    
    axes[1].plot(bucket_df["Bucket Size"], bucket_df["Precision"], marker='o', label="Precision")
    axes[1].plot(bucket_df["Bucket Size"], bucket_df["Recall"], marker='s', label="Recall")
    axes[1].plot(bucket_df["Bucket Size"], bucket_df["Fscore"], marker='^', label="Fscore")
    axes[1].set_title("PRF vs LSH Bucket Size")
    axes[1].set_xlabel("Bucket Size")
    axes[1].set_ylabel("Metric Value")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title("MinHash Compression Ratio and Approximation Accuracy")
    
    color1 = 'tab:blue'
    ax1.set_xlabel("Signature Rows Used")
    ax1.set_ylabel("Compression Ratio", color=color1)
    ax1.plot(compression_df["Signature Rows Used"], compression_df["Compression Ratio"], marker='o', color=color1, label="Compression Ratio")
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel("Accuracy", color=color2)
    ax2.plot(compression_df["Signature Rows Used"], compression_df["Accuracy"], marker='s', color=color2, label="Accuracy")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Rocchio Relevance Feedback MAP Evaluation")
    
    labels = [f"a={r['Alpha']}, b={r['Beta']}, \ng={r['Gamma']}" for _, r in map_df.iterrows()]
    
    axes[0].plot(labels, map_df["MAP Before"], marker='o', label="MAP Before")
    axes[0].plot(labels, map_df["MAP After"], marker='s', label="MAP After")
    axes[0].set_title("MAP for Different Reweighting Configs")
    axes[0].set_xlabel("Term Reweighting Settings")
    axes[0].set_ylabel("Mean Average Precision")
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend()
    
    axes[1].bar(labels, map_df["Percent Change in MAP"], color='purple')
    axes[1].set_title("Percent Change in MAP")
    axes[1].set_xlabel("Term Reweighting Settings")
    axes[1].set_ylabel("Percent Change")
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.show()

def main():
    setup_seeds()

    docs = load_corpus("corpus")
    
    if not docs:
        print("Error: No documents found in the 'corpus/' directory.")
        return

    k_shingles = 2
    
    processed_docs_words = [preprocess(doc) for doc in docs]
    
    processed_docs = [" ".join(words) for words in processed_docs_words]
    
    shingles = [set(get_k_shingles(words, k=k_shingles)) for words in processed_docs_words]

    signature, minhash_sim, vocab, num_hash = compute_minhash(shingles, docs, num_hash=100)
    compute_lsh(signature, bands=50)
    
    vectorizer, tfidf, scores, new_scores = compute_rocchio(processed_docs, docs)
    lca_scores = compute_lca(processed_docs, docs, vectorizer, tfidf, scores)
    
    jaccard_matrix = compute_jaccard(shingles, docs)
    
    bucket_df = evaluate_prf(signature, jaccard_matrix, docs, num_hash)
    compression_df = evaluate_compression(signature, jaccard_matrix, docs, vocab)
    map_df = evaluate_map(vectorizer, tfidf, docs)
    
    plot_graphs(minhash_sim, jaccard_matrix, scores, new_scores, bucket_df, compression_df, map_df, docs, shingles)
    
    width = 60
    print(f"\n{'*' * width}")
    print("                     FINAL EVALUATION REPORT                  ")
    print(f"{'*' * width}\n")
    print(f"Corpus Size:               {len(docs)} documents")
    print(f"Total Unique Tokens:       {len(vocab)}")
    print(f"Average Jaccard Sim:       {np.mean(jaccard_matrix):.3f}")
    print(f"Average MinHash Sim:       {np.mean(minhash_sim):.3f}")
    print(f"Average Original VSM Score:{np.mean(scores):.3f}")
    print(f"Average Rocchio Score:     {np.mean(new_scores):.3f}")
    print(f"Average LCA Score:         {np.mean(lca_scores):.3f}")
    print("\nLSH Configuration Details")
    print(f"   Hashes Generated:       {num_hash}")
    print(f"   Bands Evaluated:        50")
    print(f"   Rows per Band:          {num_hash // 50}")
    print(f"\n{'*' * width}\n")

if __name__ == "__main__":
    main()
