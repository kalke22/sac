import pandas as pd
import random
import graphviz
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


# Tree visualization function using graphviz
def visualize_tree(node, parent_name="Root", graph=None):
    if graph is None:
        graph = graphviz.Digraph(format="png", engine="dot")

    if node.is_leaf():
        graph.node(parent_name, label=str(node.label), shape="ellipse")
    else:
        label = (
            f"{node.feature} <= {node.median_value}"
            if node.median_value
            else str(node.feature)
        )
        graph.node(parent_name, label=label, shape="box")
        for val, child in node.children.items():
            child_name = f"{parent_name}_{val}"
            graph.edge(parent_name, child_name, label=str(val))
            visualize_tree(child, child_name, graph)

    return graph


def main():
    feature_columns = [col for col in df.columns if col not in ["Default id", "Tid"]]

    tree_root = build_tree(df, target_column="Default id", feature_columns=feature_columns)

    # Visualize the tree using graphviz
    graph = visualize_tree(tree_root)
    output_path = os.path.join(os.path.dirname(__file__), "hunts_decision_tree.png")
    graph.render(output_path, view=True, cleanup=True)
    print(f"Decision tree rendered and saved as '{output_path}.png'")


if __name__ == "__main__":
    main()
