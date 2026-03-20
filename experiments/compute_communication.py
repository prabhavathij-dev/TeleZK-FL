"""
Communication Cost Calculator (Step 37).

Calculates exact communication payload sizes for both models
and reports FP32 vs INT8 compression ratios.

Usage: python experiments/compute_communication.py
"""
import sys
import os
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mobilenetv2_2d import get_mobilenetv2_2d, count_parameters
from src.models.mobilenetv2_1d import get_mobilenetv2_1d
from src.quantization.ptq import quantize_model_delta
from src.zkp.prover import TeleZKProver
from src.quantization.lut_builder import load_lut


def compute_communication(output_dir: str = "results/logs"):
    """Compute exact communication costs for the paper."""
    print("\n" + "=" * 60)
    print("TeleZK-FL Communication Cost Analysis")
    print("=" * 60)

    results = {}

    models = [
        ("MobileNetV2-2D (CheXpert)", get_mobilenetv2_2d(pretrained=False)),
        ("MobileNetV2-1D (PTB-XL)", get_mobilenetv2_1d(12, 5)),
    ]

    for model_name, model in models:
        print(f"\n--- {model_name} ---")

        num_params = count_parameters(model)
        state_dict = model.state_dict()
        num_layers = len(state_dict)

        # FP32 cost
        fp32_bytes = sum(p.numel() * 4 for p in state_dict.values())
        fp32_mb = fp32_bytes / (1024 * 1024)

        # INT8 cost (model weights)
        int8_bytes = sum(p.numel() for p in state_dict.values())
        # INT8 overhead: per-layer scale (float32) + zero_point (int32)
        int8_overhead = num_layers * 8  # 4 bytes scale + 4 bytes zp
        int8_total = int8_bytes + int8_overhead
        int8_mb = int8_total / (1024 * 1024)

        # ZK proof overhead (SHA-256 digest per layer + final)
        proof_bytes = 32 * (num_layers + 1)  # 32 bytes per SHA-256 hash
        proof_kb = proof_bytes / 1024

        # Total TeleZK-FL payload
        telezk_bytes = int8_total + proof_bytes
        telezk_mb = telezk_bytes / (1024 * 1024)

        # Compression ratio
        compression = fp32_bytes / telezk_bytes
        reduction_pct = (1 - telezk_bytes / fp32_bytes) * 100

        entry = {
            "model": model_name,
            "num_params": num_params,
            "num_layers": num_layers,
            "fp32_bytes": int(fp32_bytes),
            "fp32_mb": float(fp32_mb),
            "int8_bytes": int(int8_total),
            "int8_mb": float(int8_mb),
            "proof_bytes": int(proof_bytes),
            "proof_kb": float(proof_kb),
            "telezk_total_bytes": int(telezk_bytes),
            "telezk_total_mb": float(telezk_mb),
            "compression_ratio": float(compression),
            "reduction_pct": float(reduction_pct),
        }
        results[model_name] = entry

        print(f"  Parameters:     {num_params:,}")
        print(f"  Layers:         {num_layers}")
        print(f"  FP32 payload:   {fp32_mb:.2f} MB ({fp32_bytes:,} bytes)")
        print(f"  INT8 payload:   {int8_mb:.2f} MB ({int8_total:,} bytes)")
        print(f"  Proof overhead: {proof_kb:.2f} KB ({proof_bytes} bytes)")
        print(f"  TeleZK total:   {telezk_mb:.2f} MB ({telezk_bytes:,} bytes)")
        print(f"  Compression:    {compression:.1f}x ({reduction_pct:.1f}% reduction)")

    # Per-round savings (10 clients)
    print("\n--- Per-Round Savings (10 clients) ---")
    for name, entry in results.items():
        fp32_round = entry["fp32_mb"] * 10
        telezk_round = entry["telezk_total_mb"] * 10
        print(f"  {name}:")
        print(f"    Standard FL: {fp32_round:.1f} MB/round")
        print(f"    TeleZK-FL:   {telezk_round:.1f} MB/round")
        print(f"    Saved:       {fp32_round - telezk_round:.1f} MB/round")

    # 50-round total
    print("\n--- 50-Round Total (10 clients) ---")
    for name, entry in results.items():
        fp32_total = entry["fp32_mb"] * 10 * 50
        telezk_total = entry["telezk_total_mb"] * 10 * 50
        print(f"  {name}:")
        print(f"    Standard FL: {fp32_total:.0f} MB total")
        print(f"    TeleZK-FL:   {telezk_total:.0f} MB total")

    print("\n  NOTE: Actual reduction is ~4x (75%). The INT8 quantization")
    print("  provides the bulk of savings. ZK proof overhead is negligible.")
    print("  Do NOT claim 18x — report 4x or 75% consistently.")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "communication_costs.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    compute_communication()
