import pandas as pd
import numpy as np

# ===========================================================
# 📂 FILE READING & INITIAL INSPECTION
# ===========================================================

# Load data (CSV, Excel, JSON)
df = pd.read_csv('data.csv')                                              # Load CSV
df = pd.read_excel('data.xlsx')                                            # Load Excel
df = pd.read_json('data.json')                                             # Load JSON

# Quick Inspection
print(df.head()); print(df.info()); print(df.describe())                   # Basic stats & info
print(df.shape)                                                            # (rows, columns)
print(df.columns.tolist())                                                 # List all column names
print(df.dtypes)                                                           # Data types of each column
print(df.isnull().sum())                                                   # Count NaNs per column
print(df.nunique())                                                        # Unique values per column
print(df.value_counts('col'))                                              # Frequency of each value

# ===========================================================
# 🔍 SELECTION & MANIPULATION
# ===========================================================

# Selecting rows and columns
cols = df[['col1', 'col2']]                                                # Select multiple columns
rows = df.iloc[0:10]                                                       # Select first 10 rows by index
filtered = df[df['age'] > 25]                                              # Select rows based on condition
cell = df.loc[0, 'col1']                                                   # Select specific cell (label-based)
cell = df.iloc[0, 0]                                                       # Select specific cell (index-based)
filtered = df.query("age > 25 and city == 'NYC'")                          # Query-based filtering
sampled = df.sample(n=5, random_state=42)                                  # Random sample of rows

# Sorting
sorted_df = df.sort_values('col', ascending=False)                         # Sort by column descending
sorted_df = df.sort_values(['col1', 'col2'], ascending=[True, False])      # Multi-column sort

# Stacking & Merging
df_stacked = pd.concat([df1, df2], axis=0)                                 # Stack vertically (rows)
df_wide = pd.concat([df1, df2], axis=1)                                    # Stack horizontally (columns)
df_merged = pd.merge(df1, df2, on='id', how='inner')                       # SQL-like join (inner/left/right/outer)

# ===========================================================
# 🛠️ DATA CLEANING & PROCESSING
# ===========================================================

# Handling Missing Values
df['col'].fillna(df['col'].mean(), inplace=True)                           # Impute by mean (one-liner)
df['col'].fillna(df['col'].median(), inplace=True)                         # Impute by median
df['col'].fillna(df['col'].mode()[0], inplace=True)                        # Impute by mode (categorical)
df.dropna(axis=0, inplace=True)                                            # Drop rows with any NaN values
df.dropna(subset=['col1', 'col2'], inplace=True)                           # Drop rows with NaN in specific columns
df.ffill(inplace=True)                                                     # Forward fill NaN values
df.bfill(inplace=True)                                                     # Backward fill NaN values

# Transformations
df['new_col'] = df['col'].apply(lambda x: x**2)                           # Apply custom function
grouped = df.groupby('category')['sales'].sum()                            # Groupby & aggregate
grouped = df.groupby('cat').agg({'sales': 'sum', 'qty': 'mean'})           # Multiple aggregations
df['cat_code'] = df['category'].astype('category').cat.codes               # Quick label encoding
df['col'] = df['col'].str.lower()                                          # String lowercase
df['col'] = df['col'].str.replace('old', 'new')                            # String replace
df['col'] = df['col'].str.strip()                                          # Strip whitespace
df['binned'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100])               # Binning/discretization
df = pd.get_dummies(df, columns=['city'], drop_first=True)                 # One-hot encode columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)                  # Rename columns
df.drop(columns=['col1', 'col2'], inplace=True)                            # Drop columns
df.drop_duplicates(inplace=True)                                           # Remove duplicate rows

# Type Conversions
df['col'] = df['col'].astype(int)                                          # Convert column type
df['date'] = pd.to_datetime(df['date_str'])                                # Parse dates

# ===========================================================
# 🔢 NUMPY ESSENTIALS
# ===========================================================

arr = np.array([1, 2, 3])                                                  # Create array
reshaped = arr.reshape(1, -1)                                              # Reshape for sklearn (2D)
mean_val = np.mean(arr); std_val = np.std(arr)                             # Basic stats
mask = arr[arr > 2]                                                        # Boolean indexing/filtering
zeros = np.zeros((3, 3)); ones = np.ones((3, 3))                           # Zero/One matrices
eye = np.eye(3)                                                            # Identity matrix
rand = np.random.rand(3, 3)                                                # Random matrix [0,1)
dot = np.dot(arr, arr)                                                     # Dot product
norm = np.linalg.norm(arr)                                                 # Vector norm (L2)
log = np.log2(arr)                                                         # Log base 2 (entropy)
unique, counts = np.unique(arr, return_counts=True)                        # Unique values & counts

# ===========================================================
# 🤖 SCIKIT-LEARN PREPROCESSING
# ===========================================================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputing (Standard approach)
imputed = SimpleImputer(strategy='mean').fit_transform(df[['num_col']])     # Impute missing with mean
imputed = SimpleImputer(strategy='most_frequent').fit_transform(df[['cat']])# Impute categorical

# Scaling & Encoding
scaled = StandardScaler().fit_transform(df[['age', 'salary']])             # Standardize (mean=0, std=1)
scaled = MinMaxScaler().fit_transform(df[['age', 'salary']])               # Normalize to [0, 1]
encoded = OneHotEncoder().fit_transform(df[['gender']]).toarray()           # One-hot encode
le = LabelEncoder().fit_transform(df['target'])                            # Encode target labels

# Pipeline (all-in-one)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
processor = ColumnTransformer([('num', StandardScaler(), ['age']), ('cat', OneHotEncoder(), ['city'])])

# ===========================================================
# ⛏️ DATA MINING (DM) ESSENTIALS
# ===========================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.preprocessing import label_binarize

# --- Decision Trees ---
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)         # Train decision tree
y_pred = dt.predict(X_test)                                                # Predict
print(accuracy_score(y_test, y_pred))                                      # Accuracy

# --- Ensemble Methods ---
bag = BaggingClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)     # Bagging
ada = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)    # AdaBoost
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)# Random Forest

# --- Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)                      # K-Means clustering
labels = kmeans.labels_                                                    # Cluster labels
centers = kmeans.cluster_centers_                                          # Cluster centroids

# --- Classification Metrics ---
print(accuracy_score(y_test, y_pred))                                      # Accuracy = (TP+TN)/(TP+TN+FP+FN)
print(precision_score(y_test, y_pred, average='weighted'))                  # Precision = TP/(TP+FP)
print(recall_score(y_test, y_pred, average='weighted'))                     # Recall = TP/(TP+FN)
print(f1_score(y_test, y_pred, average='weighted'))                         # F1 = 2*(P*R)/(P+R)
print(confusion_matrix(y_test, y_pred))                                    # Confusion matrix
print(classification_report(y_test, y_pred))                               # Full report

# --- ROC & AUC ---
y_bin = label_binarize(y_test, classes=[0, 1, 2])                          # Binarize for multiclass ROC
fpr, tpr, _ = roc_curve(y_bin[:, 0], y_proba[:, 0])                       # ROC curve (per class)
roc_auc = auc(fpr, tpr)                                                    # AUC score

# --- Apriori / Association Rules (manual) ---
from itertools import combinations
support = lambda itemset, txns: sum(1 for t in txns if itemset.issubset(t)) / len(txns)
freq_items = {frozenset([i]) for t in transactions for i in t}             # C1 candidates
pairs = [frozenset(c) for c in combinations(sorted(items), 2)]             # Generate candidate pairs

# --- Entropy & Information Gain (ID3) ---
entropy = lambda probs: -sum(p * np.log2(p) for p in probs if p > 0)       # Shannon entropy
info_gain = lambda parent_ent, children: parent_ent - sum(w * entropy(c) for w, c in children)
gini = lambda probs: 1 - sum(p**2 for p in probs)                          # Gini impurity

# ===========================================================
# 🔎 INFORMATION RETRIEVAL (IR) ESSENTIALS
# ===========================================================

from collections import Counter
import math

# --- Term Frequency (TF) ---
# TF(t, d) = count(t in d) / total_terms_in_d
tf = lambda term, doc: doc.count(term) / len(doc)                          # Term frequency

# --- Inverse Document Frequency (IDF) ---
# IDF(t) = log(N / df(t))  where N = total docs, df(t) = docs containing t
idf = lambda term, docs: math.log(len(docs) / sum(1 for d in docs if term in d))

# --- TF-IDF ---
# TF-IDF(t, d) = TF(t, d) * IDF(t)
tfidf = lambda term, doc, docs: tf(term, doc) * idf(term, docs)            # TF-IDF score

# --- BM25 ---
# BM25(t, d) = IDF(t) * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * |d| / avgdl))
k1 = 1.5; b = 0.75                                                        # BM25 parameters
avgdl = np.mean([len(d) for d in docs])                                    # Average document length

# --- Boolean Retrieval ---
# AND: set(doc1_terms) & set(doc2_terms)
# OR:  set(doc1_terms) | set(doc2_terms)
# NOT: set(all_terms) - set(doc_terms)
bool_and = lambda q_terms, doc: all(t in doc for t in q_terms)             # Boolean AND query
bool_or = lambda q_terms, doc: any(t in doc for t in q_terms)              # Boolean OR query

# --- Cosine Similarity ---
# cos(A, B) = (A · B) / (||A|| * ||B||)
cosine_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Jaccard Similarity ---
# J(A, B) = |A ∩ B| / |A ∪ B|
jaccard = lambda a, b: len(a & b) / len(a | b)                            # Jaccard similarity (sets)

# --- Tokenization & Text Processing ---
tokens = "hello world foo bar".lower().split()                             # Basic tokenization
vocab = set(tokens)                                                        # Vocabulary
bow = Counter(tokens)                                                      # Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_matrix = TfidfVectorizer().fit_transform(["doc1 text", "doc2 text"]) # TF-IDF vectorizer

# --- Inverted Index ---
inv_index = {}                                                             # Build inverted index
for doc_id, doc in enumerate(docs):
    for term in doc:
        inv_index.setdefault(term, set()).add(doc_id)

# --- Precision & Recall (IR) ---
# Precision@k = relevant_in_top_k / k
# Recall@k    = relevant_in_top_k / total_relevant
precision_at_k = lambda retrieved, relevant, k: len(set(retrieved[:k]) & relevant) / k
recall_at_k = lambda retrieved, relevant, k: len(set(retrieved[:k]) & relevant) / len(relevant)
