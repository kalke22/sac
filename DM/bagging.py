import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

def gini(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

class TreeNode:
    def __init__(self, depth=0, max_depth=3):
        self.depth = depth
        self.max_depth = max_depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X, y):
        if len(set(y)) == 1:
            self.value = y[0]
            return
        if self.depth >= self.max_depth or len(y) <= 2:
            self.value = Counter(y).most_common(1)[0][0]
            return
        n_samples, n_features = X.shape
        best_gini = 1.0
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                g = (sum(left_mask)/n_samples)*gini(y[left_mask]) + \
                    (sum(right_mask)/n_samples)*gini(y[right_mask])
                if g < best_gini:
                    best_gini = g
                    self.feature = feature
                    self.threshold = t
        if self.feature is None:
            self.value = Counter(y).most_common(1)[0][0]
            return
        left_mask = X[:, self.feature] <= self.threshold
        right_mask = X[:, self.feature] > self.threshold
        self.left = TreeNode(depth=self.depth+1, max_depth=self.max_depth)
        self.left.fit(X[left_mask], y[left_mask])
        self.right = TreeNode(depth=self.depth+1, max_depth=self.max_depth)
        self.right.fit(X[right_mask], y[right_mask])

    def predict(self, X):
        if self.value is not None:
            return np.array([self.value]*len(X))
        left_mask = X[:, self.feature] <= self.threshold
        right_mask = X[:, self.feature] > self.threshold
        y_pred = np.empty(X.shape[0], dtype=int)
        if sum(left_mask) > 0:
            y_pred[left_mask] = self.left.predict(X[left_mask])
        if sum(right_mask) > 0:
            y_pred[right_mask] = self.right.predict(X[right_mask])
        return y_pred

np.random.seed(42)
n_estimators = 10
models = []

for i in range(n_estimators):
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    tree = TreeNode(max_depth=3)
    tree.fit(X_sample, y_sample)
    models.append(tree)

all_preds = np.array([model.predict(X_test) for model in models])
final_preds = []

for i in range(len(X_test)):
    votes = Counter(all_preds[:, i])
    final_preds.append(votes.most_common(1)[0][0])

final_preds = np.array(final_preds)
accuracy = np.sum(final_preds == y_test) / len(y_test)

print("First 10 Predictions:", final_preds[:10])
print("Accuracy:", accuracy)

"""Fully libraries"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

base_classifier = DecisionTreeClassifier(random_state=42)

bagging_model = BaggingClassifier(
    estimator=base_classifier,
    n_estimators=10,
    random_state=42
)

bagging_model.fit(X_train, y_train)

y_pred = bagging_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Bagging Classifier Accuracy:", accuracy)

"""Only DT as library function"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

np.random.seed(42)
n_estimators = 10
estimators = []

for i in range(n_estimators):
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_sample = X_train[indices]
    y_sample = y_train[indices]

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_sample, y_sample)
    estimators.append(tree)

all_preds = np.array([tree.predict(X_test) for tree in estimators])

final_preds = []
for i in range(len(X_test)):
    votes = Counter(all_preds[:, i])
    final_preds.append(votes.most_common(1)[0][0])
final_preds = np.array(final_preds)

accuracy = accuracy_score(y_test, final_preds)
print("First 10 Predictions:", final_preds[:10])
print("Manual Bagging Accuracy:", accuracy)
