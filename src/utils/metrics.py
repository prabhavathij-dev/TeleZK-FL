"""
Metrics Module for TeleZK-FL.

Provides evaluation metrics (AUC) and communication cost calculations
for federated learning experiments.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score


def compute_auc_per_class(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: str = "cpu",
) -> Tuple[Dict[str, float], float]:
    """Compute per-class ROC-AUC on the test set.

    Args:
        model: Trained model in eval mode.
        test_loader: DataLoader for test data.
        class_names: List of class names.
        device: Device to run inference on.

    Returns:
        Tuple of (per_class_auc dict, mean_auc).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            all_preds.append(output.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_class_auc = {}
    valid_aucs = []

    for i, name in enumerate(class_names):
        try:
            # Check if class has both positive and negative samples
            unique_labels = np.unique(all_labels[:, i])
            if len(unique_labels) < 2:
                # Only one class present - AUC undefined, use 0.5
                per_class_auc[name] = 0.5
            else:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                per_class_auc[name] = float(auc)
                valid_aucs.append(auc)
        except Exception as e:
            print(f"  Warning: Could not compute AUC for {name}: {e}")
            per_class_auc[name] = 0.5

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return per_class_auc, mean_auc


def compute_communication_cost(
    model_state_dict: dict,
    bits: int = 32,
) -> Tuple[float, float]:
    """Calculate the size of a model update in bytes and MB.

    Args:
        model_state_dict: Model state dict (or delta dict).
        bits: Bits per parameter (32 for FP32, 8 for INT8).

    Returns:
        Tuple of (size_bytes, size_mb).
    """
    total_params = 0
    for name, param in model_state_dict.items():
        if isinstance(param, torch.Tensor):
            total_params += param.numel()
        else:
            total_params += np.prod(np.array(param).shape)

    bytes_per_param = bits / 8
    size_bytes = total_params * bytes_per_param

    # Add small overhead for scales and zero_points in INT8 case
    if bits == 8:
        num_layers = len(model_state_dict)
        # Each layer has a scale (float32) and zero_point (int)
        overhead = num_layers * (4 + 4)
        size_bytes += overhead

    size_mb = size_bytes / (1024 * 1024)
    return size_bytes, size_mb
