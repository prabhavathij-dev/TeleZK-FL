"""
FL+DP Baseline Runner (Step 30).

Runs FedAvg WITH Differential Privacy (gradient clipping + Gaussian noise).
Produces the 'FL + DP (epsilon=2)' row in Table 1.

DP configuration:
    - Clipping norm C = 1.0
    - Privacy budget epsilon = 2.0
    - Failure probability delta = 1e-5
    - Noise sigma = C * sqrt(2 * ln(1.25/delta)) / epsilon

Usage:
    python experiments/run_dp_baseline.py
    python experiments/run_dp_baseline.py --dataset chexpert --partition iid --seed 42
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fl.trainer import run_federated_experiment


CONFIGS = {
    ("chexpert", "iid"): "config/chexpert_iid.yaml",
    ("chexpert", "noniid"): "config/chexpert_noniid.yaml",
    ("ptbxl", "iid"): "config/ptbxl_iid.yaml",
    ("ptbxl", "noniid"): "config/ptbxl_noniid.yaml",
}
SEEDS = [42, 123, 456]


def run_single_dp(dataset: str, partition: str, seed: int):
    """Run a single FL+DP experiment."""
    key = (dataset, partition)
    if key not in CONFIGS:
        raise ValueError(f"Unknown config: {key}.")

    config_path = CONFIGS[key]
    print(f"\n{'='*60}")
    print(f"FL+DP Baseline: {dataset}/{partition} seed={seed}")
    print(f"  epsilon=2.0, delta=1e-5, clip_norm=1.0")
    print(f"{'='*60}")

    return run_federated_experiment(config_path, seed_override=seed, mode="dp")


def run_all_dp():
    """Run all 12 FL+DP experiments."""
    total = len(CONFIGS) * len(SEEDS)
    print(f"\nRunning {total} FL+DP baseline experiments")
    print("=" * 60)

    for i, ((dataset, partition), config_path) in enumerate(CONFIGS.items()):
        for j, seed in enumerate(SEEDS):
            run_num = i * len(SEEDS) + j + 1
            print(f"\n[{run_num}/{total}] {dataset}/{partition} seed={seed}")
            try:
                run_federated_experiment(config_path, seed_override=seed, mode="dp")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nAll {total} FL+DP experiments complete!")


def main():
    parser = argparse.ArgumentParser(description="FL+DP Baseline Runner")
    parser.add_argument("--dataset", choices=["chexpert", "ptbxl"], default=None)
    parser.add_argument("--partition", choices=["iid", "noniid"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Run all 12 configs")
    args = parser.parse_args()

    if args.all or (args.dataset is None and args.seed is None):
        run_all_dp()
    else:
        dataset = args.dataset or "chexpert"
        partition = args.partition or "iid"
        seed = args.seed or 42
        run_single_dp(dataset, partition, seed)


if __name__ == "__main__":
    main()
