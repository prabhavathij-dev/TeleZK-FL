"""
Federated Learning Server for TeleZK-FL v2.

Handles ZK proof verification, weighted FedAvg aggregation of
dequantized client updates, and global model evaluation.
"""

import copy
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

from src.quantization.ptq import dequantize_model_delta
from src.zkp.prover import ProofResult
from src.zkp.verifier import TeleZKVerifier
from src.utils.metrics import compute_auc_per_class


class FLServer:
    """Federated Learning server with ZK verification and FedAvg.

    Args:
        global_model: The global model to aggregate updates into.
        verifier: TeleZKVerifier instance for proof verification.
    """

    def __init__(self, global_model: nn.Module, verifier: TeleZKVerifier):
        self.global_model = global_model
        self.verifier = verifier

    def verify_and_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_proofs: List[ProofResult],
        client_sizes: List[int],
        client_scales: List[Dict[str, torch.Tensor]],
        client_zero_points: List[Dict[str, int]],
    ) -> Tuple[int, int, float]:
        """Verify proofs and aggregate valid client updates.

        Uses weighted FedAvg where weights are proportional to
        each client's dataset size.

        Args:
            client_updates: List of INT8 quantized delta dicts.
            client_proofs: List of ProofResult objects.
            client_sizes: List of sample counts per client.
            client_scales: List of quantization scale dicts.
            client_zero_points: List of quantization zero_point dicts.

        Returns:
            Tuple of (num_valid, num_total, aggregation_time_seconds).
        """
        agg_start = time.perf_counter()

        # Step 1: Verify each client's proof (skip if no verifier)
        valid_indices = []
        if self.verifier is not None:
            for i, (proof, update) in enumerate(zip(client_proofs, client_updates)):
                is_valid, v_time = self.verifier.verify_proof(proof, update)
                if is_valid:
                    valid_indices.append(i)
                else:
                    print(f"  Server: Client {i} proof FAILED verification")
        else:
            # No verifier (baseline/DP mode): accept all updates
            valid_indices = list(range(len(client_proofs)))

        num_valid = len(valid_indices)
        num_total = len(client_proofs)

        if num_valid == 0:
            print("  Server: No valid proofs! Skipping aggregation.")
            return 0, num_total, time.perf_counter() - agg_start

        # Step 2: Dequantize valid updates
        dequantized_updates = []
        valid_sizes = []
        for i in valid_indices:
            deq = dequantize_model_delta(
                client_updates[i],
                client_scales[i],
                client_zero_points[i],
            )
            dequantized_updates.append(deq)
            valid_sizes.append(client_sizes[i])

        # Step 3: Weighted FedAvg aggregation
        total_samples = sum(valid_sizes)
        global_state = self.global_model.state_dict()

        for name in dequantized_updates[0]:
            if name in global_state:
                weighted_sum = torch.zeros_like(global_state[name], dtype=torch.float32)
                for update, n_k in zip(dequantized_updates, valid_sizes):
                    if name in update:
                        weight = n_k / total_samples
                        weighted_sum += weight * update[name]

                global_state[name] = global_state[name].float() + weighted_sum

        self.global_model.load_state_dict(global_state)

        agg_time = time.perf_counter() - agg_start
        return num_valid, num_total, agg_time

    def get_global_weights(self) -> dict:
        """Return a deep copy of the global model's state dict."""
        return copy.deepcopy(self.global_model.state_dict())

    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """Evaluate the global model on the test set.

        Args:
            test_loader: DataLoader for test data.
            class_names: List of class names for AUC reporting.

        Returns:
            Tuple of (per_class_auc dict, mean_auc).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_model.to(device)
        return compute_auc_per_class(
            self.global_model, test_loader, class_names, device
        )
