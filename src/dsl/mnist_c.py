#!/usr/bin/env python

import argparse
import os
import random
import urllib.request
import numpy as np
import torch
from torch.utils.data import random_split
from numpy import save
import zipfile


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_and_extract_mnist_c(destination):
    url = "https://zenodo.org/record/3239543/files/mnist_c.zip"
    zip_path = os.path.join(destination, "mnist_c.zip")
    extract_path = os.path.join(destination, "mnist_c")
    
    if not os.path.exists(extract_path):
        print("Downloading MNIST-C dataset...")
        os.makedirs(destination, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        
        print("Extraction complete.")
        os.remove(zip_path)
    else:
        print("MNIST-C dataset already downloaded.")


def load_mnist_c_data(dataset_dir, valid_ratio=0.2):

    corruption_types = [
        "shear", "scale", "rotate", "translate"
    ]
    
    test_corruptions = corruption_types
    
    all_features, all_labels = [], []
    test_features, test_labels = [], []
    for i,corruption in enumerate(corruption_types):
        images = np.load(os.path.join(dataset_dir, f"{corruption}/train_images.npy"))
        labels = np.load(os.path.join(dataset_dir, f"{corruption}/train_labels.npy"))
        all_features.append(images)
        all_labels.append(labels)

        #subsample from class
        instances_per_class = int(len(all_features[i])/len(corruption_types))
        all_features[i] = all_features[i][:instances_per_class]
        all_labels[i] = all_labels[i][:instances_per_class]

    for i, corruption in enumerate(test_corruptions):
        images = np.load(os.path.join(dataset_dir, f"{corruption}/test_images.npy"))
        labels = np.load(os.path.join(dataset_dir, f"{corruption}/test_labels.npy"))
        test_features.append(images)
        test_labels.append(labels)
        

    all_features = np.concatenate(all_features, axis=0)  # Shape: (15 * 10k, 28, 28)
    all_labels = np.concatenate(all_labels, axis=0)  # Shape: (15 * 10k,)

    test_features = np.concatenate(test_features, axis=0)  # Shape: (15 * 10k, 28, 28)
    test_labels = np.concatenate(test_labels, axis=0)  # Shape: (15 * 10k,) 

    # Shuffle before splitting
    indices = np.arange(len(all_features))
    np.random.shuffle(indices)
    all_features, all_labels = all_features[indices], all_labels[indices]
    indices = np.arange(len(test_features))
    np.random.shuffle(indices)    
    test_features, test_labels = test_features[indices], test_labels[indices]
    
    # Split into train/valid
    nb_train = int((1.0 - valid_ratio) * (len(all_features)))
    train_X, dev_X = all_features[:nb_train], all_features[nb_train:]
    train_y, dev_y = all_labels[:nb_train], all_labels[nb_train:]
    
    return train_X, train_y, dev_X, dev_y, test_features, test_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data", help="Path to save MNIST-C dataset.")
    opt = parser.parse_args()
    
    configure_seed(seed=42)
    dataset_dir = os.path.join(opt.path, "mnist_c")
    
    # Download and extract dataset
    download_and_extract_mnist_c(opt.path)
    
    # Load and preprocess data
    train_X, train_y, dev_X, dev_y, test_X, test_y = load_mnist_c_data(dataset_dir)
    
    # Print some stats
    print(f"Train: {len(train_X)} samples")
    print(f"Dev: {len(dev_X)} samples")
    print(f"Test: {len(test_X)} samples")
    
    # Save as NumPy arrays
    os.makedirs(opt.path, exist_ok=True)
    save(os.path.join(opt.path, "train_features.npy"), train_X)
    save(os.path.join(opt.path, "train_labels.npy"), train_y)
    save(os.path.join(opt.path, "dev_features.npy"), dev_X)
    save(os.path.join(opt.path, "dev_labels.npy"), dev_y)
    save(os.path.join(opt.path, "test_features.npy"), test_X)
    save(os.path.join(opt.path, "test_labels.npy"), test_y)   
    print("MNIST-C dataset saved successfully.")


if __name__ == "__main__":
    main()
