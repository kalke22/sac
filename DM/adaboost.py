from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# Bagging Classifier
# ==========================================

print("=" * 50)
print("BAGGING CLASSIFIER")
print("=" * 50)

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    random_state=42,
)

bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred_bagging):.4f}")
print(f"\nFirst 10 Predictions: {y_pred_bagging[:10]}")
print(f"First 10 Actual:      {y_test[:10]}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_bagging, target_names=iris.target_names)}")

# ==========================================
# AdaBoost Classifier
# ==========================================

print("=" * 50)
print("ADABOOST CLASSIFIER")
print("=" * 50)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42,
)

adaboost_model.fit(X_train, y_train)
y_pred_adaboost = adaboost_model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred_adaboost):.4f}")
print(f"\nFirst 10 Predictions: {y_pred_adaboost[:10]}")
print(f"First 10 Actual:      {y_test[:10]}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_adaboost, target_names=iris.target_names)}")

# ==========================================
# Comparison
# ==========================================

print("=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"Bagging Accuracy:  {accuracy_score(y_test, y_pred_bagging):.4f}")
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred_adaboost):.4f}")
