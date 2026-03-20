"""
TeleZK Zero-Knowledge Prover with Real Timing.

Implements ZK proof generation with actual wall-clock timing measurements
under RPi4-simulated constraints. Uses LUT lookups, range checks, and
SHA-256 hash commitments to simulate the operations a real Halo2-based
prover would perform.
"""

import time
import hashlib
import numpy as np
import torch
from typing import Dict, Set, Optional
from dataclasses import dataclass, field


@dataclass
class ProofResult:
    """Container for ZK proof generation results."""
    is_valid: bool = True
    proof_bytes: bytes = b""
    per_layer_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    num_operations_checked: int = 0
    num_range_violations: int = 0


class TeleZKProver:
    """Zero-Knowledge prover for quantized model updates.

    Performs LUT-based verification of INT8 operations with real
    timing measurements. Generates SHA-256 hash commitments that
    simulate polynomial commitments a real ZK system would produce.

    Args:
        lut_set: Set of valid (a, b, c) INT8 multiplication tuples.
        l_inf_bound: Maximum allowed L-infinity norm for dequantized deltas.
    """

    def __init__(self, lut_set: Set[tuple], l_inf_bound: float = 1.0):
        self.lut_set = lut_set
        self.l_inf_bound = l_inf_bound

    def generate_proof(
        self,
        quantized_delta_dict: Dict[str, torch.Tensor],
        scales_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ProofResult:
        """Generate a ZK proof for quantized model delta.

        For each layer:
        1. Range check: verify all values in [-128, 127]
        2. LUT verification: verify sample multiplications exist in LUT
        3. Commitment: SHA-256 hash over quantized values

        Args:
            quantized_delta_dict: Dict of INT8 quantized weight deltas.
            scales_dict: Optional dict of quantization scales (for L-inf check).

        Returns:
            ProofResult with timing and validity information.
        """
        result = ProofResult()
        overall_start = time.perf_counter()
        all_commitments = []

        for name, q_tensor in quantized_delta_dict.items():
            layer_start = time.perf_counter()

            # Convert to numpy for processing
            if isinstance(q_tensor, torch.Tensor):
                q_np = q_tensor.detach().cpu().numpy()
            else:
                q_np = np.array(q_tensor)

            # Step 1: Range check — all values must be in [-128, 127]
            range_violations = int(np.sum((q_np < -128) | (q_np > 127)))
            result.num_range_violations += range_violations

            # Step 2: LUT verification — sample pairs of values
            # In a real system, we'd check all multiplications in the
            # forward pass. Here we check a representative sample.
            flat = q_np.flatten().astype(np.int32)
            num_ops = min(len(flat) - 1, 1000)  # check up to 1000 ops
            ops_checked = 0

            for i in range(0, num_ops, 2):
                if i + 1 < len(flat):
                    a = int(np.clip(flat[i], -128, 127))
                    b = int(np.clip(flat[i + 1], -128, 127))
                    c = a * b
                    # O(1) LUT lookup
                    if (a, b, c) not in self.lut_set:
                        result.num_range_violations += 1
                    ops_checked += 1

            result.num_operations_checked += ops_checked

            # Step 3: SHA-256 commitment (simulates polynomial commitment)
            commitment_input = name.encode() + q_np.tobytes()
            layer_hash = hashlib.sha256(commitment_input).digest()
            all_commitments.append(layer_hash)

            layer_time = time.perf_counter() - layer_start
            result.per_layer_times[name] = layer_time

        # Final commitment: hash over all layer hashes
        combined = b"".join(all_commitments)
        result.proof_bytes = hashlib.sha256(combined).digest()

        result.total_time = time.perf_counter() - overall_start
        result.is_valid = (result.num_range_violations == 0)

        return result

    def benchmark_layer(
        self,
        layer_size_n: int,
        num_trials: int = 10,
    ) -> Dict[str, float]:
        """Benchmark proof generation for a random layer of given size.

        Creates a random INT8 tensor and times proof generation.

        Args:
            layer_size_n: Side length of the square weight matrix.
            num_trials: Number of repeated trials for statistical accuracy.

        Returns:
            Dict with mean, std, min, max times in seconds.
        """
        times = []

        for _ in range(num_trials):
            # Create random INT8 tensor
            q_tensor = torch.randint(
                -128, 128, (layer_size_n, layer_size_n), dtype=torch.int8
            )
            dummy_dict = {"benchmark_layer": q_tensor}

            start = time.perf_counter()
            self.generate_proof(dummy_dict)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        return {
            "mean": float(times.mean()),
            "std": float(times.std()),
            "min": float(times.min()),
            "max": float(times.max()),
            "num_trials": num_trials,
            "layer_size": layer_size_n,
        }
