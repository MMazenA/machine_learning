import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature_name, probability=0, children=None, label="") -> None:
        self.feature_name = feature_name
        self.label = label
        self.probability = probability
        self.children = children

    def add_child(self, node):
        self.children.append(node)

    def __str__(self):
        return (
            f"Label: {self.feature_name} {self.label}\nProbability: {self.probability}"
        )


class C45:
    def __init__(self, dataset: pd.DataFrame, theta: int) -> None:
        self.raw_dataset = dataset
        self.theta = theta
        self.trees = [Node("root")] * dataset.shape[1]

    def calc_ig(node: Node):
        information_gain = 0
        for i in range(len(node.children)):
            single_gain = node.probability * np.log2(node.probability)
            node.probability += single_gain
        return information_gain


def main():
    dataset = pd.read_csv("adult.csv")
    print(dataset.shape)


if __name__ == "__main__":
    main()
