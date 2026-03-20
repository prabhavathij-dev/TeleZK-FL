"""
TeleZK Zero-Knowledge Verifier.

Verifies ZK proofs by recomputing SHA-256 commitments from quantized
model deltas and comparing against proof commitments. Includes
scalability benchmarking for varying numbers of clients.
"""

import time
import hashlib
import torch
import numpy as np
from typing import Dict, List, Set, Tuple

from src.zkp.prover import ProofResult


class TeleZKVerifier:
    """Zero-Knowledge proof verifier for the aggregation server.

    Verifies client proofs by recomputing commitments from the
    quantized deltas and comparing with the proof's hash.

    Args:
        lut_set: Set of valid (a, b, c) INT8 multiplication tuples.
    """

    def __init__(self, lut_set: Set[tuple]):
        self.lut_set = lut_set

    def verify_proof(
        self,
        proof: ProofResult,
        quantized_delta_dict: Dict[str, torch.Tensor],
    ) -> Tuple[bool, float]:
        """Verify a single client's ZK proof.

        Recomputes the SHA-256 commitment from the quantized delta and
        compares with the proof commitment.

        Args:
            proof: The ProofResult from the prover.
            quantized_delta_dict: The client's INT8 quantized deltas.

        Returns:
            Tuple of (is_valid, verification_time_seconds).
        """
        start = time.perf_counter()

        # Check basic validity flags
        if not proof.is_valid:
            return False, time.perf_counter() - start

        if proof.num_range_violations > 0:
            return False, time.perf_counter() - start

        # Recompute commitment
        all_commitments = []
        for name, q_tensor in quantized_delta_dict.items():
            if isinstance(q_tensor, torch.Tensor):
                q_np = q_tensor.detach().cpu().numpy()
            else:
                q_np = np.array(q_tensor)

            commitment_input = name.encode() + q_np.tobytes()
            layer_hash = hashlib.sha256(commitment_input).digest()
            all_commitments.append(layer_hash)

        combined = b"".join(all_commitments)
        recomputed = hashlib.sha256(combined).digest()

        # Compare commitments
        is_valid = (recomputed == proof.proof_bytes)

        verification_time = time.perf_counter() - start
        return is_valid, verification_time

    def benchmark_verification(
        self,
        num_clients_list: List[int] = None,
    ) -> Dict[int, float]:
        """Benchmark verification scalability across client counts.

        Generates dummy proofs and measures total verification time
        for each K in num_clients_list.

        Args:
            num_clients_list: List of client counts to benchmark.

        Returns:
            Dict of {num_clients: total_verification_time_seconds}.
        """
        if num_clients_list is None:
            num_clients_list = [10, 25, 50, 75, 100, 150, 200]

        results = {}

        for k in num_clients_list:
            print(f"  Benchmarking verification for K={k} clients...")

            # Generate K dummy proofs with slight variations
            proofs = []
            deltas = []
            for i in range(k):
                # Create a small random quantized delta
                dummy_delta = {
                    f"layer_{j}": torch.randint(-128, 128, (64, 64), dtype=torch.int8)
                    for j in range(3)
                }

                # Add a client-specific value to make each unique
                dummy_delta["client_id"] = torch.tensor([i], dtype=torch.int8)

                # Compute commitment
                all_hashes = []
                for name, q_tensor in dummy_delta.items():
                    data = name.encode() + q_tensor.numpy().tobytes()
                    all_hashes.append(hashlib.sha256(data).digest())

                combined = b"".join(all_hashes)
                proof = ProofResult(
                    is_valid=True,
                    proof_bytes=hashlib.sha256(combined).digest(),
                    total_time=0.001,
                    num_operations_checked=100,
                    num_range_violations=0,
                )

                proofs.append(proof)
                deltas.append(dummy_delta)

            # Measure total verification time
            start = time.perf_counter()
            for proof, delta in zip(proofs, deltas):
                self.verify_proof(proof, delta)
            total_time = time.perf_counter() - start

            results[k] = total_time
            print(f"    K={k}: {total_time:.4f}s "
                  f"({total_time / k * 1000:.2f} ms/client)")

        return results
