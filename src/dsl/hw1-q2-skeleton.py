#!/usr/bin/env python

import argparse
import random
import os
from itertools import count
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(path, normalisation=False):
    """
    path: location of MNIST-C data: expects a folder with np arrays. See hw1-dataload.py
    feature_rep: a function or None. Use it to transform the binary pixel
        representation into something more interesting in Q 2.2a
    bias: whether to add a bias term as an extra feature dimension
    """

    train_X = np.load(os.path.join(path, "train_features.npy"))
    train_y = np.load(os.path.join(path, "train_labels.npy"))
    dev_X = np.load(os.path.join(path, "dev_features.npy"))
    dev_y = np.load(os.path.join(path, "dev_labels.npy"))
    test_X = np.load(os.path.join(path, "test_features.npy"))
    test_y = np.load(os.path.join(path, "test_labels.npy"))

    train_X = train_X.reshape(train_X.shape[0], -1) / 255
    dev_X = dev_X.reshape(dev_X.shape[0], -1) / 255
    test_X = test_X.reshape(test_X.shape[0], -1) / 255

    if normalisation:
        mean = np.mean(train_X.reshape(-1, 1))
        std = np.std(train_X.reshape(-1, 1))
        print("normalisation: ", mean, std)
        train_X = (train_X - mean) / std
        dev_X = (dev_X - mean) / std
        test_X = (test_X - mean) / std

    return {
        "train": (train_X, train_y),
        "dev": (dev_X, dev_y),
        "test": (test_X, test_y),
    }


def relu(z):
    return np.clip(z, 0, None)


def relu_prime(z):
    return z > 0


f_derivatives = {relu: relu_prime}


def softmax(z, axis=None):
    raw = np.exp(z)
    return raw / raw.sum(axis=axis)


def stable_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in tqdm(zip(X, y)):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]

        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Question 2.1 a
        raise NotImplementedError


class LogisticRegression(LinearModel):

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        l2_penalty (float): BONUS
        """
        # Question 2.2 b
        raise NotImplementedError


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size, layers):
        in_sizes = [n_features] + [hidden_size[i] for i in range(layers)]
        out_sizes = [hidden_size[i] for i in range(layers)] + [n_classes]
        self.weights = [
            np.random.normal(size=(in_size, out_size))
            for in_size, out_size in zip(in_sizes, out_sizes)
        ]
        self.biases = [np.zeros(out_size) for out_size in out_sizes]
        self.activations = [relu for i in range(layers)] + [stable_softmax]

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X, y):
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        raise NotImplementedError


def plot(epochs, valid_accs, test_accs, save_fig=None):
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label="validation")
    plt.plot(epochs, test_accs, label="test")
    plt.legend()
    plt.show()
    if save_fig:
        plt.savefig(save_fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["perceptron", "logistic_regression", "mlp"],
        help="Which model should the script run?",
    )
    parser.add_argument(
        "--data", default="data", help="Path to (MNIST-C) dataset."
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""",
    )
    parser.add_argument("--hidden_sizes", type=list, default=[100], help="List of hidden sizes for each layer")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--save_fig", default=None, help="Path to save the plot.")
    parser.add_argument("--normalisation", default=False, help="Normalisation or not")
    opt = parser.parse_args()

    configure_seed(seed=42)

    data = load_data(opt.data, opt.normalisation)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == "perceptron":
        model = Perceptron(n_classes, n_feats)
    elif opt.model == "logistic_regression":
        model = LogisticRegression(n_classes, n_feats)
    else:
        # Q3
        model = MLP(n_classes, n_feats, opt.hidden_sizes, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print("Training epoch {}".format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == "mlp" or opt.model == "logistic_regression":
            model.train_epoch(train_X, train_y, learning_rate=opt.learning_rate)
        else:
            model.train_epoch(train_X, train_y)
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        print(
            "dev: {:.4f} | test: {:.4f}".format(
                valid_accs[-1],
                test_accs[-1],
            )
        )

    # plot
    plot(epochs, valid_accs, test_accs, opt.save_fig)


if __name__ == "__main__":
    main()
