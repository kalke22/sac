import pandas as pd
import numpy as np
import random
import os
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, classification_report,
)
import matplotlib.pyplot as plt


# ===========================================================
# 1. DATA LOADING & PREPROCESSING
# ===========================================================

# --- Load CSV data for Hunt's and ID3 ---
data_path = os.path.join(os.path.dirname(__file__), "data.csv")
df = pd.read_csv(data_path)
df["Annual Income"] = (
    df["Annual Income"]
    .astype(str)
    .str.replace("K", "", regex=False)
    .str.replace(" ", "", regex=False)
    .astype(int)
    * 1000
)

# --- Load Iris for Bagging, AdaBoost, Metrics ---
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)
print(f"CSV Dataset shape: {df.shape}")
print(f"CSV Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nIris Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(iris.target_names)} classes")
print(f"Train/Test split: {len(X_train)}/{len(X_test)}")


# ===========================================================
# 2. HUNT'S ALGORITHM (Random Feature Selection)
# ===========================================================

class HuntsNode:
    def __init__(self, feature=None, median_value=None, label=None):
        self.feature = feature
        self.median_value = median_value
        self.children = {}
        self.label = label

    def is_leaf(self):
        return self.label is not None


def hunts_build_tree(df, target_column, feature_columns):
    if len(df[target_column].unique()) == 1:
        return HuntsNode(label=df[target_column].mode()[0])
    if not feature_columns:
        return HuntsNode(label=df[target_column].mode()[0])

    feature = random.choice(feature_columns)
    node = HuntsNode(feature=feature)
    remaining_features = [col for col in feature_columns if col != feature]

    if pd.api.types.is_numeric_dtype(df[feature]):
        median_value = df[feature].median()
        node.median_value = median_value
        node.children["<= " + str(median_value)] = hunts_build_tree(
            df[df[feature] <= median_value], target_column, remaining_features
        )
        node.children["> " + str(median_value)] = hunts_build_tree(
            df[df[feature] > median_value], target_column, remaining_features
        )
    else:
        for val in df[feature].unique():
            node.children[val] = hunts_build_tree(
                df[df[feature] == val], target_column, remaining_features
            )
    return node


def print_tree(node, indent=""):
    if node.is_leaf():
        print(f"{indent}Leaf: {node.label}")
        return
    if node.median_value is not None:
        print(f"{indent}[Numeric Split] {node.feature} <= {node.median_value}")
    else:
        print(f"{indent}[Categorical Split] {node.feature}")
    for val, child in node.children.items():
        print(f"{indent}--> {val}:")
        print_tree(child, indent + "    ")


print("\n" + "=" * 60)
print("HUNT'S ALGORITHM - Decision Tree")
print("=" * 60)

hunts_features = [col for col in df.columns if col not in ["Default id", "Tid"]]
hunts_tree = hunts_build_tree(df, target_column="Default id", feature_columns=hunts_features)
print_tree(hunts_tree)


# ===========================================================
# 3. ID3 ALGORITHM (Entropy-based Feature Selection)
# ===========================================================

# Entropy: H(S) = -Σ p(x) * log2(p(x))
def entropy(df, target_column):
    counts = df[target_column].value_counts()
    probs = counts / len(df)
    return -sum(probs * np.log2(probs))


# Information Gain: IG(S, A) = H(S) - Σ (|Sv|/|S|) * H(Sv)
def information_gain(df, feature, target_column):
    total_entropy = entropy(df, target_column)
    weighted_entropy = 0
    for value in df[feature].unique():
        subset = df[df[feature] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset, target_column)
    return total_entropy - weighted_entropy


def best_feature(df, feature_columns, target_column):
    gains = {f: information_gain(df, f, target_column) for f in feature_columns}
    return max(gains, key=gains.get)


class ID3Node:
    def __init__(self, feature=None, value=None, label=None):
        self.feature = feature
        self.value = value
        self.children = {}
        self.label = label

    def is_leaf(self):
        return self.label is not None


def id3(df, target_column, feature_columns):
    if len(df[target_column].unique()) == 1:
        return ID3Node(label=df[target_column].mode()[0])
    if not feature_columns:
        return ID3Node(label=df[target_column].mode()[0])

    feature = best_feature(df, feature_columns, target_column)
    node = ID3Node(feature=feature)
    remaining_features = [col for col in feature_columns if col != feature]

    if pd.api.types.is_numeric_dtype(df[feature]):
        median_value = df[feature].median()
        node.value = f"{feature} <= {median_value}"
        node.children["<= " + str(median_value)] = id3(
            df[df[feature] <= median_value], target_column, remaining_features
        )
        node.children["> " + str(median_value)] = id3(
            df[df[feature] > median_value], target_column, remaining_features
        )
    else:
        for val in df[feature].unique():
            node.children[val] = id3(
                df[df[feature] == val], target_column, remaining_features
            )
    return node


def print_id3_tree(node, indent=""):
    if node.is_leaf():
        print(f"{indent}Leaf: {node.label}")
        return
    if node.value:
        print(f"{indent}[Numeric Split] {node.value}")
    else:
        print(f"{indent}[Categorical Split] {node.feature}")
    for val, child in node.children.items():
        print(f"{indent}--> {val}:")
        print_id3_tree(child, indent + "    ")


print("\n" + "=" * 60)
print("ID3 ALGORITHM - Decision Tree (data.csv)")
print("=" * 60)

id3_features = [col for col in df.columns if col not in ["Default id", "Tid"]]
id3_tree = id3(df, target_column="Default id", feature_columns=id3_features)
print_id3_tree(id3_tree)

# Tennis dataset
tennis_path = os.path.join(os.path.dirname(__file__), "tennis.csv")
if os.path.exists(tennis_path):
    tennis_df = pd.read_csv(tennis_path)
    tennis_features = [col for col in tennis_df.columns if col != "Play"]
    tennis_tree = id3(tennis_df, target_column="Play", feature_columns=tennis_features)
    print("\n" + "=" * 60)
    print("ID3 ALGORITHM - Decision Tree (tennis.csv)")
    print("=" * 60)
    print_id3_tree(tennis_tree)


# ===========================================================
# 4. BAGGING CLASSIFIER
# ===========================================================

print("\n" + "=" * 60)
print("BAGGING CLASSIFIER (Iris)")
print("=" * 60)

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10, random_state=42,
)
bagging_model.fit(X_train, y_train)
y_pred_bag = bagging_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_bag):.4f}")
print(f"First 10 Predictions: {y_pred_bag[:10]}")


# ===========================================================
# 5. ADABOOST CLASSIFIER
# ===========================================================

print("\n" + "=" * 60)
print("ADABOOST CLASSIFIER (Iris)")
print("=" * 60)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50, learning_rate=1.0, random_state=42,
)
adaboost_model.fit(X_train, y_train)
y_pred_ada = adaboost_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada):.4f}")
print(f"First 10 Predictions: {y_pred_ada[:10]}")


# ===========================================================
# 6. METRICS & COMPARISON
# ===========================================================

print("\n" + "=" * 60)
print("METRICS - BAGGING")
print("=" * 60)
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_bag):.4f}")
# Precision = TP / (TP + FP)
print(f"Precision: {precision_score(y_test, y_pred_bag, average='weighted'):.4f}")
# Recall = TP / (TP + FN)
print(f"Recall:    {recall_score(y_test, y_pred_bag, average='weighted'):.4f}")
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
print(f"F1 Score:  {f1_score(y_test, y_pred_bag, average='weighted'):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_bag)}")
print(f"\n{classification_report(y_test, y_pred_bag, target_names=iris.target_names)}")

print("=" * 60)
print("METRICS - ADABOOST")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_ada):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_ada, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_ada, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_ada, average='weighted'):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_ada)}")
print(f"\n{classification_report(y_test, y_pred_ada, target_names=iris.target_names)}")

print("=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Bagging Accuracy:  {accuracy_score(y_test, y_pred_bag):.4f}")
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred_ada):.4f}")


# ===========================================================
# 7. VISUALIZATION (Confusion Matrix Plot - matplotlib)
# ===========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, title in zip(
    axes, [y_pred_bag, y_pred_ada], ["Bagging", "AdaBoost"]
):
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{title} - Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(iris.target_names))
    ax.set_xticks(ticks); ax.set_xticklabels(iris.target_names, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(iris.target_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("all_confusion_matrices.png", dpi=150)
plt.show()
print("\nConfusion matrix plots saved as 'all_confusion_matrices.png'")
