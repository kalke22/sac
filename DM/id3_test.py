import pandas as pd
import numpy as np
import graphviz
import os


# Load and process data
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


# Entropy calculation
def entropy(df, target_column):
    counts = df[target_column].value_counts()
    probs = counts / len(df)
    return -sum(probs * np.log2(probs))


# Information gain calculation
def information_gain(df, feature, target_column):
    total_entropy = entropy(df, target_column)

    values = df[feature].unique()

    weighted_entropy = 0
    for value in values:
        subset = df[df[feature] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset, target_column)

    return total_entropy - weighted_entropy


# Best feature selection
def best_feature(df, feature_columns, target_column):
    gains = {
        feature: information_gain(df, feature, target_column)
        for feature in feature_columns
    }
    return max(gains, key=gains.get)


# Node class
class ID3Node:
    def __init__(self, feature=None, value=None, label=None):
        self.feature = feature
        self.value = value
        self.children = {}
        self.label = label

    def is_leaf(self):
        return self.label is not None


# ID3 algorithm
def id3(df, target_column, feature_columns):
    # If the target column is pure, return a leaf node
    if len(df[target_column].unique()) == 1:
        return ID3Node(label=df[target_column].mode()[0])

    # If no features left, return leaf with majority class
    if not feature_columns:
        return ID3Node(label=df[target_column].mode()[0])

    feature = best_feature(df, feature_columns, target_column)
    node = ID3Node(feature=feature)

    if pd.api.types.is_numeric_dtype(df[feature]):
        median_value = df[feature].median()
        left_df = df[df[feature] <= median_value]
        right_df = df[df[feature] > median_value]

        node.value = f"{feature} <= {median_value}"

        remaining_features = [col for col in feature_columns if col != feature]
        node.children["<= " + str(median_value)] = id3(
            left_df, target_column, remaining_features
        )
        node.children["> " + str(median_value)] = id3(
            right_df, target_column, remaining_features
        )
    else:
        unique_vals = df[feature].unique()
        for val in unique_vals:
            subset = df[df[feature] == val]
            remaining_features = [col for col in feature_columns if col != feature]
            node.children[val] = id3(subset, target_column, remaining_features)

    return node


# Tree visualization function using graphviz
def visualize_id3_tree(node, parent_name="Root", graph=None):
    if graph is None:
        graph = graphviz.Digraph(format="png", engine="dot")

    if node.is_leaf():
        graph.node(parent_name, label=str(node.label), shape="ellipse")
    else:
        if node.value:
            label = node.value
        else:
            label = str(node.feature)
        graph.node(parent_name, label=label, shape="box")

        for val, child in node.children.items():
            child_name = f"{parent_name}_{val}"
            graph.edge(parent_name, child_name, label=str(val))
            visualize_id3_tree(child, child_name, graph)

    return graph


def main():
    feature_columns = [col for col in df.columns if col not in ["Default id", "Tid"]]

    tree_root = id3(df, target_column="Default id", feature_columns=feature_columns)

    # Visualize the tree using graphviz
    graph = visualize_id3_tree(tree_root)
    output_path = os.path.join(os.path.dirname(__file__), "id3_decision_tree")
    graph.render(output_path, view=True, cleanup=True)
    print(f"Decision tree rendered and saved as '{output_path}.png'")

    # Tennis dataset
    tennis_path = os.path.join(os.path.dirname(__file__), "tennis.csv")
    tennis_df = pd.read_csv(tennis_path)

    tennis_features = [col for col in tennis_df.columns if col != "Play"]
    tennis_tree = id3(tennis_df, target_column="Play", feature_columns=tennis_features)

    graph_tennis = visualize_id3_tree(tennis_tree)
    tennis_output_path = os.path.join(
        os.path.dirname(__file__), "id3_tennis_decision_tree"
    )
    graph_tennis.render(tennis_output_path, view=True, cleanup=True)
    print(f"Tennis decision tree rendered and saved as '{tennis_output_path}.png'")


if __name__ == "__main__":
    main()
