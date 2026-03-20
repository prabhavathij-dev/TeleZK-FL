"""
Scalability Benchmark for TeleZK-FL v2.

Measures verification time as the number of clients K increases
from 10 to 200 to demonstrate linear scalability of the TeleZK-FL
aggregator verification process.
"""

import os
import json
import numpy as np
from typing import Dict, List

from src.quantization.lut_builder import build_int8_multiplication_lut, load_lut
from src.zkp.verifier import TeleZKVerifier


def run_scalability_benchmark(
    output_dir: str = "results/logs",
    client_counts: List[int] = None,
) -> Dict:
    """Run the verification scalability benchmark.

    Measures total verification time for K clients' proofs,
    demonstrating approximately linear scaling.

    Args:
        output_dir: Directory to save results JSON.
        client_counts: List of K values to benchmark.

    Returns:
        Results dict.
    """
    if client_counts is None:
        client_counts = [10, 25, 50, 75, 100, 125, 150, 175, 200]

    print("\n" + "=" * 60)
    print("TeleZK-FL Verification Scalability Benchmark")
    print("=" * 60)

    # Setup
    lut_path = os.path.join("data", "mul_lut_int8.npy")
    if os.path.exists(lut_path):
        _, lut_set = load_lut(lut_path)
    else:
        _, lut_set = build_int8_multiplication_lut(lut_path)

    verifier = TeleZKVerifier(lut_set)

    # Run benchmark
    print("\nBenchmarking verification time vs number of clients...")
    timing_results = verifier.benchmark_verification(client_counts)

    # Format results
    results = {
        "scalability": [],
        "is_linear": True,
    }

    print(f"\n{'Clients K':<12} {'Total Time (s)':<18} {'Per Client (ms)':<18}")
    print("-" * 48)

    prev_ratio = None
    for k in client_counts:
        total_time = timing_results[k]
        per_client = total_time / k * 1000

        entry = {
            "num_clients": k,
            "total_verification_time_s": float(total_time),
            "per_client_time_ms": float(per_client),
        }
        results["scalability"].append(entry)

        print(f"{k:<12} {total_time:>14.4f}    {per_client:>14.2f}")

    # Check linearity: verify per-client time roughly constant
    per_client_times = [e["per_client_time_ms"] for e in results["scalability"]]
    cv = np.std(per_client_times) / np.mean(per_client_times) if np.mean(per_client_times) > 0 else 0
    results["coefficient_of_variation"] = float(cv)
    results["is_linear"] = cv < 0.5  # CV < 50% suggests roughly linear

    print(f"\nPer-client time CV: {cv:.3f} "
          f"({'linear' if results['is_linear'] else 'non-linear'})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "scalability_benchmark.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")

    return results
