"""
Data Partitioning Module for Federated Learning.

Supports IID (uniform random) and non-IID (Dirichlet) partitioning of
a dataset across K FL clients. Returns lists of torch Subset objects.
"""

import numpy as np
from typing import List
from collections import Counter

import torch
from torch.utils.data import Dataset, Subset


def partition_iid(
    dataset: Dataset,
    num_clients: int = 10,
    seed: int = 42,
) -> List[Subset]:
    """Partition dataset into IID subsets for each client.

    Randomly shuffles all indices and splits into num_clients equal parts.

    Args:
        dataset: The full dataset to partition.
        num_clients: Number of FL clients.
        seed: Random seed for reproducibility.

    Returns:
        List of Subset objects, one per client.
    """
    rng = np.random.RandomState(seed)
    num_samples = len(dataset)
    indices = rng.permutation(num_samples)

    # Split into roughly equal parts
    split_indices = np.array_split(indices, num_clients)

    subsets = [Subset(dataset, idx.tolist()) for idx in split_indices]
    return subsets


def partition_dirichlet(
    dataset: Dataset,
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Subset]:
    """Partition dataset using Dirichlet distribution (non-IID).

    For multi-label datasets, uses the argmax of the label vector as
    the "primary" class for partitioning purposes.

    Lower alpha = more heterogeneous distribution across clients.
    alpha=0.5 is standard in FL literature.

    Args:
        dataset: The full dataset to partition.
        num_clients: Number of FL clients.
        alpha: Dirichlet concentration parameter.
        seed: Random seed for reproducibility.

    Returns:
        List of Subset objects, one per client.
    """
    rng = np.random.RandomState(seed)
    num_samples = len(dataset)

    # Extract labels to determine primary class without loading images
    if hasattr(dataset, 'df'):
        # CheXpert: read labels from dataframe
        raw_labels = dataset.df[dataset.df.columns[-5:]].values
        primary_labels = np.argmax(raw_labels, axis=1)
    elif hasattr(dataset, 'labels'):
        # PTB-XL: labels already in memory
        primary_labels = np.argmax(dataset.labels, axis=1)
    else:
        # Fallback: sample a subset to avoid loading entire dataset
        primary_labels = []
        for i in range(min(num_samples, 2000)):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            if label.ndim > 0 and len(label) > 1:
                primary_labels.append(int(np.argmax(label)))
            else:
                primary_labels.append(int(label))
        
        primary_labels = np.array(primary_labels)
        if num_samples > 2000:
            extra = rng.choice(primary_labels, size=num_samples - 2000)
            primary_labels = np.concatenate([primary_labels, extra])

    # Initialize client index lists
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute indices using Dirichlet
    for c in range(num_classes):
        class_indices = np.where(primary_labels == c)[0]
        rng.shuffle(class_indices)

        # Draw proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to actual counts
        # Use cumulative sum approach to avoid rounding issues
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
        splits = np.split(class_indices, proportions[:-1])

        for k in range(num_clients):
            if k < len(splits):
                client_indices[k].extend(splits[k].tolist())

    # Shuffle each client's indices
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    subsets = [Subset(dataset, indices) for indices in client_indices]
    return subsets


def print_partition_stats(
    client_datasets: List[Subset],
    num_classes: int,
    class_names: List[str] = None,
) -> None:
    """Print statistics about the data partition across clients.

    Shows sample count and class distribution for each client.

    Args:
        client_datasets: List of Subset objects, one per client.
        num_classes: Total number of classes.
        class_names: Optional list of class names for display.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print("\n" + "=" * 60)
    print("Data Partition Statistics")
    print("=" * 60)

    for k, subset in enumerate(client_datasets):
        # Count class distribution without loading images where possible
        class_counts = Counter()
        dataset = subset.dataset
        
        if hasattr(dataset, 'df'):
            # CheXpert dataframe
            raw_labels = dataset.df.iloc[subset.indices][dataset.df.columns[-5:]].values
            for row in raw_labels:
                for j, val in enumerate(row):
                    if val > 0.5:
                        class_counts[j] += 1
        elif hasattr(dataset, 'labels'):
            # PTB-XL labels
            raw_labels = dataset.labels[subset.indices]
            for row in raw_labels:
                for j, val in enumerate(row):
                    if val > 0.5:
                        class_counts[j] += 1
        else:
            # Fallback
            for idx in subset.indices:
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.numpy()
                if label.ndim > 0 and len(label) > 1:
                    for j in range(len(label)):
                        if label[j] > 0.5:
                            class_counts[j] += 1
                else:
                    class_counts[int(label)] += 1

        dist_str = " | ".join(
            f"{class_names[j]}: {class_counts.get(j, 0)}"
            for j in range(num_classes)
        )
        print(f"  Client {k:2d}: {len(subset):5d} samples  [{dist_str}]")

    total = sum(len(s) for s in client_datasets)
    print(f"\n  Total samples: {total}")
    print("=" * 60 + "\n")
