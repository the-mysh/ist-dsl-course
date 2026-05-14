#!/usr/bin/env python
import os
import torch
import numpy as np
import argparse
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import os
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F

plt.style.use('seaborn-v0_8')


class MNISTC(Dataset):
    def __init__(self, root, mode="train", transform=None):
        """
        Args:
            root (str): Path to the extracted MNIST-C dataset.
            transform (callable, optional): Transformations to apply.
        """
        self.mode = mode

        data_path = os.path.join(root, f'{self.mode}_features.npy')
        labels_path = os.path.join(root, f'{self.mode}_labels.npy')

        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Convert to tensor and apply transformations
        image = self.transform(image)

        return image, label


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


## Question 3.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)
        """
        super(LogisticRegression, self).__init__()

        self.weight = nn.Parameter(torch.zeros(n_classes, n_features))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        return x @ self.weight.T + self.bias


## Question 3.2
class FeedforwardNetwork(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_features: int,
        hidden_sizes: list[int],
        n_layers: int,
        activation_type: str,
        dropout: float,
        **kwargs
    ):
        """
        n_classes (int)
        n_features (int)
        hidden_sizes (list) Note: can also be an int
        layers (int)
        activation_type (str)
        dropout (float): dropout probability
        """
        super(FeedforwardNetwork, self).__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = n_layers * [hidden_sizes]
        elif not isinstance(hidden_sizes, list):
            raise TypeError(f"hidden_sizes: expected an int or list of ints, got {type(hidden_sizes)}")
        elif (nh := len(hidden_sizes)) != n_layers:
            raise ValueError(f"Specified {n_layers} layers, but {nh} hidden layer sizes")

        sizes = [n_features] + hidden_sizes + [n_classes]
        layers = []
        for size_in, size_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(size_in, size_out, bias=True))
        self.layers = nn.ModuleList(layers)

        match activation_type:
            case "tanh":
                activation_fn = nn.Tanh()
            case "relu":
                activation_fn = nn.ReLU
            case _:
                raise ValueError(f"Activation type '{activation_type}' not recognised")

        self.activations = [activation_fn() for _ in range(len(self.layers) - 1)] + [nn.Identity()]

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


## Question 3.3
class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.1, channels=1):
        super().__init__()
        # Needs to be implemented!

    def forward(self, input):
        raise NotImplementedError


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """

    optimizer.zero_grad()       # clear old gradients

    y_est = model(X)            # evaluate -> logits (n_examples x n_classes)
    loss = criterion(y_est, y)  # compare to ground truth, calculate loss
    loss.backward()             # compute gradients
    optimizer.step()            # update weights

    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    return predicted_labels


def evaluate(model, dataloader, modeltype="cnn"):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    n_correct = 0
    n_possible = 0
    y_pred = []
    y_true = []
    model.eval()
    for X, y in dataloader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        if modeltype != "cnn":
            X = X.reshape(X.shape[0], -1)
        y_hat = predict(model, X)
        y_pred.extend(y_hat)
        y_true.extend(y)
        n_correct += (y == y_hat).sum().item()
        n_possible += float(y.shape[0])

    model.train()

    return n_correct / n_possible


def plot(epochs, plottable, ylabel="", save_path=None):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.show()
    if save_path:
        plt.savefig(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["logistic_regression", "mlp", "cnn"],
        help="Which model should the script run?",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Size of training batch."
    )
    parser.add_argument("--path", default="data", help="Path where MNIST-C dataset is saved")
    parser.add_argument("--hidden_sizes", nargs="*", type=int, default=[200, 20])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--l2_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--activation", choices=["tanh", "relu"], default="relu")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--save-path", default=None, help="Template path to save plots.")
    opt = parser.parse_args()

    configure_seed(seed=42)

    transform_gray = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    dataset_dir = opt.path
    test_dataset = MNISTC(root=dataset_dir, mode="test", transform=transform_gray)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    # Get dataset info
    n_classes = len(set(test_dataset.labels))  # Should be 10
    n_feats = test_dataset.data.shape[1]  # Should be 28 (image height)

    print(f"Classes: {n_classes}, Feature Size: {n_feats}")

    train_dataset = MNISTC(root=dataset_dir, mode="train",transform=transform_gray)
    valid_ratio = 0.2

    nb_train = int((1.0 - valid_ratio) * len(train_dataset))
    nb_valid = int(valid_ratio * len(train_dataset))
    train_dataset, dev_dataset = torch.utils.data.dataset.random_split(
        train_dataset, [nb_train, nb_valid]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=True)

    # initialize the model
    if opt.model == "logistic_regression":
        model = LogisticRegression(n_classes, n_feats * n_feats)
    elif opt.model == "cnn":
        model = CNN(n_classes, dropout=opt.dropout, channels=1)
    else:
        model = FeedforwardNetwork(
            n_classes,
            n_feats * n_feats,
            opt.hidden_sizes,
            opt.layers,
            opt.activation,
            opt.dropout,
        )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)

    train_mean_losses = []
    valid_accs = []
    train_losses = []

    checkpoint_dir = Path(".")/f"checkpoints_{datetime.now().strftime("%Y%m%d-%H%M%S")}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for ii in epochs:
        print("Training epoch {}".format(ii))
        for X_batch, y_batch in tqdm(train_dataloader):
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            if opt.model != "cnn":
                X_batch = X_batch.reshape(opt.batch_size, -1)
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print("Training loss: %.4f" % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accuracy = evaluate(
            model, dev_dataloader, modeltype=opt.model
        )
        print("Valid acc: %.4f" % (valid_accuracy))
        if len(valid_accs) < 1 or valid_accuracy > max(valid_accs):
            path_to_model = str(checkpoint_dir / (opt.model + "_epoch_" + str(ii) + ".pt"))
            torch.save(
                {
                    "epoch": ii,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": mean_loss,
                },
                path_to_model,
            )
        valid_accs.append(valid_accuracy)

    ## load best model
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        "Final Test acc: %.4f"
        % (evaluate(model, test_dataloader, modeltype=opt.model))
    )

    # plot
    plot(
        epochs,
        train_mean_losses,
        ylabel="Loss",
        save_path=opt.save_path.format("loss") if opt.save_path else None
    )
    plot(
        epochs,
        valid_accs,
        ylabel="Accuracy",
        save_path=opt.save_path.format("accuracy") if opt.save_path else None
    )


if __name__ == "__main__":
    main()
