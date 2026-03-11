import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ==========================================
# Load dataset and train a classifier
# ==========================================

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# ==========================================
# 1. ACCURACY
# ==========================================
# Formula:
#   Accuracy = (TP + TN) / (TP + TN + FP + FN)
#   OR equivalently:
#   Accuracy = Number of Correct Predictions / Total Number of Predictions
#
# Where:
#   TP = True Positives  (correctly predicted positive)
#   TN = True Negatives  (correctly predicted negative)
#   FP = False Positives (incorrectly predicted positive, Type I error)
#   FN = False Negatives (incorrectly predicted negative, Type II error)

acc = accuracy_score(y_test, y_pred)
print("=" * 50)
print("1. ACCURACY")
print("=" * 50)
print(f"Accuracy: {acc:.4f}")

# ==========================================
# 2. CONFUSION MATRIX
# ==========================================
# The confusion matrix is a table that describes the performance of a classifier.
# For binary classification:
#
#                  Predicted Positive   Predicted Negative
#   Actual Positive       TP                  FN
#   Actual Negative       FP                  TN
#
# For multiclass: C[i][j] = number of samples with true label i predicted as label j

cm = confusion_matrix(y_test, y_pred)
print("\n" + "=" * 50)
print("2. CONFUSION MATRIX")
print("=" * 50)
print(f"\n{cm}")
print(f"\nLabels: {iris.target_names}")

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax)
tick_marks = np.arange(len(iris.target_names))
ax.set_xticks(tick_marks)
ax.set_xticklabels(iris.target_names, rotation=45, ha="right")
ax.set_yticks(tick_marks)
ax.set_yticklabels(iris.target_names)

# Add text annotations to the confusion matrix
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

ax.set_ylabel("True Label", fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# ==========================================
# 3. PRECISION
# ==========================================
# Formula:
#   Precision = TP / (TP + FP)
#
# Precision answers: "Of all instances predicted as positive, how many are actually positive?"
# High precision = low false positive rate

prec = precision_score(y_test, y_pred, average="weighted")
prec_per_class = precision_score(y_test, y_pred, average=None)
print("\n" + "=" * 50)
print("3. PRECISION")
print("=" * 50)
print(f"Weighted Precision: {prec:.4f}")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {prec_per_class[i]:.4f}")

# ==========================================
# 4. RECALL (Sensitivity / True Positive Rate)
# ==========================================
# Formula:
#   Recall = TP / (TP + FN)
#
# Recall answers: "Of all actual positive instances, how many did we correctly predict?"
# Also known as Sensitivity or True Positive Rate (TPR)
# High recall = low false negative rate

rec = recall_score(y_test, y_pred, average="weighted")
rec_per_class = recall_score(y_test, y_pred, average=None)
print("\n" + "=" * 50)
print("4. RECALL (Sensitivity / TPR)")
print("=" * 50)
print(f"Weighted Recall: {rec:.4f}")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {rec_per_class[i]:.4f}")

# ==========================================
# 5. F1 SCORE
# ==========================================
# Formula:
#   F1 = 2 * (Precision * Recall) / (Precision + Recall)
#
# F1 Score is the harmonic mean of Precision and Recall.
# It provides a balance between Precision and Recall.
# Range: 0 (worst) to 1 (best)
# Use F1 when you need a balance between Precision and Recall,
# especially with imbalanced datasets.

f1 = f1_score(y_test, y_pred, average="weighted")
f1_per_class = f1_score(y_test, y_pred, average=None)
print("\n" + "=" * 50)
print("5. F1 SCORE")
print("=" * 50)
print(f"Weighted F1 Score: {f1:.4f}")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {f1_per_class[i]:.4f}")

# ==========================================
# Full Classification Report
# ==========================================
print("\n" + "=" * 50)
print("FULL CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ==========================================
# 6. ROC CURVE (One-vs-Rest for multiclass)
# ==========================================
# ROC = Receiver Operating Characteristic
#
# The ROC curve plots:
#   X-axis: False Positive Rate (FPR) = FP / (FP + TN)
#   Y-axis: True Positive Rate (TPR) = TP / (TP + FN)  (same as Recall)
#
# AUC (Area Under the ROC Curve):
#   AUC = 1.0 means perfect classifier
#   AUC = 0.5 means random classifier (diagonal line)
#   AUC < 0.5 means worse than random
#
# For multiclass, we use One-vs-Rest (OvR) strategy

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#e74c3c", "#2ecc71", "#3498db"]

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{iris.target_names[i]} (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
ax.set_title("ROC Curve (One-vs-Rest)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("\nROC curve plot saved as 'roc_curve.png'")

# ==========================================
# 7. PRECISION-RECALL CURVE (One-vs-Rest for multiclass)
# ==========================================
# The Precision-Recall curve plots:
#   X-axis: Recall = TP / (TP + FN)
#   Y-axis: Precision = TP / (TP + FP)
#
# This curve is especially useful for imbalanced datasets
# where the positive class is rare.
#
# A good classifier has a curve that stays close to the top-right corner
# (high precision and high recall simultaneously).

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(n_classes):
    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test_bin[:, i], y_proba[:, i]
    )
    ax.plot(recall_vals, precision_vals, color=colors[i], lw=2,
            label=f"{iris.target_names[i]}")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve (One-vs-Rest)", fontsize=14, fontweight="bold")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.show()
print("\nPrecision-Recall curve plot saved as 'precision_recall_curve.png'")
