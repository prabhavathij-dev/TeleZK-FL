"""
Federated Learning Client for TeleZK-FL v2.

Handles local training on client data partitions, weight delta computation,
INT8 quantization, and ZK proof generation. Works with real MobileNetV2
models and multi-label classification.
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Optional

from src.quantization.ptq import quantize_model_delta
from src.zkp.prover import TeleZKProver, ProofResult


class FLClient:
    """Federated Learning client with quantization and ZK proof support.

    Args:
        client_id: Unique client identifier.
        model: A copy of the global model (MobileNetV2-2D or 1D).
        train_dataset: This client's data partition (a torch Subset).
        config: Dict of training hyperparameters.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_dataset: Subset,
        config: dict,
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_dataset = train_dataset
        self.config = config

        batch_size = config.get("training", {}).get("batch_size", 32)
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        # Use GPU if available — RPi4 comment left for reference only
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_local(
        self,
        global_weights: dict,
        local_epochs: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """Train locally and return weight delta.

        Args:
            global_weights: State dict from the global model.
            local_epochs: Number of local training epochs.

        Returns:
            Dict of {layer_name: delta_tensor} weight updates.
        """
        # Load global weights then move to device BEFORE snapshotting initial weights.
        # CRITICAL: snapshot AFTER .to(device) so delta computation stays on same device.
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        self.model.train()

        # Snapshot initial weights (already on correct device)
        initial_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        # Setup optimizer and loss
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        criterion = nn.BCELoss()  # models output Sigmoid, so BCELoss is correct
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if self.client_id == 0:
            print(f"\n  [Client 0] device={self.device}, lr={lr}")

        # Training loop
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.float().to(self.device)  # ensure float32

                optimizer.zero_grad()
                try:
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"  Client {self.client_id}: Batch error: {e}")
                    continue

            if self.client_id == 0 and num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"  [Client 0] Epoch {epoch+1}/{local_epochs} loss={avg_loss:.4f}")

        # Compute weight deltas (ensure they're on CPU for quantization)
        delta = {}
        for name, param in self.model.named_parameters():
            if name in initial_weights:
                delta[name] = (param.data - initial_weights[name]).cpu()

        return delta

    def quantize_and_prove(
        self,
        delta: Dict[str, torch.Tensor],
        prover: TeleZKProver,
    ) -> Tuple[dict, ProofResult, dict, dict]:
        """Quantize weight delta and generate ZK proof.

        Args:
            delta: Dict of FP32 weight deltas.
            prover: TeleZKProver instance.

        Returns:
            Tuple of (quantized_delta, proof, scales, zero_points).
        """
        # Quantize delta to INT8
        quantized_delta, scales, zero_points = quantize_model_delta(delta)

        # Generate ZK proof
        proof = prover.generate_proof(quantized_delta, scales)

        return quantized_delta, proof, scales, zero_points

    @property
    def num_samples(self) -> int:
        """Number of training samples this client has."""
        return len(self.train_dataset)
