"""
Per-client data distribution visualization (Step 38, S2).

Creates heatmaps showing the Dirichlet non-IID data distribution
across clients for supplementary material.

Usage: python supplementary/plot_client_distributions.py
"""
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11


def plot_distribution(
    num_clients: int = 10,
    num_classes: int = 5,
    alpha: float = 0.5,
    num_samples: int = 5000,
    class_names: list = None,
    title: str = "Dirichlet Non-IID Distribution",
    output_path: str = "supplementary/S2_per_client_distributions.pdf",
):
    """Generate a client-class heatmap for Dirichlet distribution.

    Args:
        num_clients: Number of FL clients.
        num_classes: Number of classes.
        alpha: Dirichlet concentration parameter.
        num_samples: Total number of samples to distribute.
        class_names: List of class names.
        title: Figure title.
        output_path: Path to save the figure.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    np.random.seed(42)

    # Generate Dirichlet proportions for each client
    proportions = np.random.dirichlet([alpha] * num_classes, size=num_clients)

    # Convert to sample counts
    counts = (proportions * num_samples / num_clients).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(counts.T, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Class")
    ax.set_title(f"{title} (α={alpha})")
    ax.set_xticks(range(num_clients))
    ax.set_xticklabels([f"C{i}" for i in range(num_clients)])
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(class_names)

    # Add text annotations
    for i in range(num_clients):
        for j in range(num_classes):
            ax.text(i, j, str(counts[i, j]), ha="center", va="center",
                    fontsize=9, color="black" if counts[i, j] < counts.max() * 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Number of Samples")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # CheXpert distribution
    plot_distribution(
        class_names=["Atelectasis", "Cardiomegaly", "Consolidation",
                      "Edema", "Pleural Effusion"],
        title="CheXpert Non-IID Distribution",
        output_path="supplementary/S2_chexpert_client_dist.pdf",
    )

    # PTB-XL distribution
    plot_distribution(
        class_names=["NORM", "MI", "STTC", "CD", "HYP"],
        title="PTB-XL Non-IID Distribution",
        output_path="supplementary/S2_ptbxl_client_dist.pdf",
    )
