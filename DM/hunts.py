import pandas as pd
import random
import os


# Load and process data
data_path = os.path.join(os.path.dirname(__file__), "data.csv")
df = pd.read_csv(data_path)
df["Annual Income"] = (
    df["Annual Income"].str.replace("K", "").str.replace(" ", "").astype(int) * 1000
)


# Node class
class Node:
    def __init__(self, feature=None, median_value=None, label=None):
        self.feature = feature
        self.median_value = median_value
        self.children = {}
        self.label = label

    def is_leaf(self):
        return self.label is not None


# Build tree function
def build_tree(df, target_column, feature_columns):
    # If target column is pure, return a leaf node
    if len(df[target_column].unique()) == 1:
        return Node(label=df[target_column].mode()[0])

    # If no features left, return leaf with majority class
    if not feature_columns:
        return Node(label=df[target_column].mode()[0])

    # Randomly select a feature
    feature = random.choice(feature_columns)
    node = Node(feature=feature)

    # Remove the selected feature from the list of available features
    remaining_features = [col for col in feature_columns if col != feature]

    # Numeric split
    if pd.api.types.is_numeric_dtype(df[feature]):
        median_value = df[feature].median()
        left_df = df[df[feature] <= median_value]
        right_df = df[df[feature] > median_value]
        node.median_value = median_value
        node.children["<= " + str(median_value)] = build_tree(
            left_df, target_column, remaining_features
        )
        node.children["> " + str(median_value)] = build_tree(
            right_df, target_column, remaining_features
        )
    else:
        # Categorical split
        for val in df[feature].unique():
            node.children[val] = build_tree(
                df[df[feature] == val], target_column, remaining_features
            )

    return node


# Print tree function
def print_tree(node, indent=""):
    if node.is_leaf():
        print(f"{indent}Leaf: {node.label}")
        return

    # Numeric split
    if node.median_value is not None:
        print(f"{indent}[Numeric Split] {node.feature} <= {node.median_value}")
        for val, child in node.children.items():
            print(f"{indent}--> {val}:")
            print_tree(child, indent + "    ")

    # Categorical split
    elif node.children:
        print(f"{indent}[Categorical Split] {node.feature}")
        for val, child in node.children.items():
            print(f"{indent}--> {val}:")
            print_tree(child, indent + "    ")


def main():
    feature_columns = [col for col in df.columns if col not in ["Default id", "Tid"]]

    tree_root = build_tree(df, target_column="Default id", feature_columns=feature_columns)

    print("=== Hunt's Algorithm - Decision Tree ===\n")
    print_tree(tree_root)


if __name__ == "__main__":
    main()
