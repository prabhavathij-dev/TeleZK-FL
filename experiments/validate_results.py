"""
Results Validation Script (Step 33).

Loads all experiment logs and validates that results are consistent
and within expected ranges. Prints a summary table.

Usage: python experiments/validate_results.py
"""
import sys
import os
import json
import glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_logs(pattern: str):
    """Load all JSON logs matching a glob pattern."""
    files = glob.glob(pattern)
    results = []
    for f in files:
        with open(f, "r") as fh:
            results.append(json.load(fh))
    return results


def extract_final_auc(logs):
    """Extract final-round mean AUC from a list of experiment logs."""
    aucs = []
    for log in logs:
        rounds = log.get("rounds", [])
        if rounds:
            aucs.append(rounds[-1].get("mean_auc", 0))
    return aucs


def validate_results(log_dir: str = "results/logs"):
    """Run all validation checks and print summary."""
    print("\n" + "=" * 70)
    print("TeleZK-FL Results Validation")
    print("=" * 70)

    all_ok = True
    issues = []

    # ── 1. Check for FL experiment logs ──────────────────────────
    print("\n[1] Checking experiment logs exist...")

    configs = [
        ("chexpert", "iid"), ("chexpert", "dirichlet"),
        ("ptbxl", "iid"), ("ptbxl", "dirichlet"),
    ]

    for dataset, partition in configs:
        pattern = os.path.join(log_dir, f"{dataset}_{partition}_*.json")
        files = glob.glob(pattern)
        count = len(files)
        status = "OK" if count >= 3 else f"WARN ({count}/3 seeds)"
        if count < 3:
            issues.append(f"Missing logs for {dataset}/{partition}: {count}/3")
        print(f"  {dataset}/{partition}: {count} logs  [{status}]")

    # ── 2. Check benchmark logs ──────────────────────────────────
    print("\n[2] Checking benchmark logs...")
    benchmarks = ["latency_benchmark.json", "energy_benchmark.json",
                   "scalability_benchmark.json"]
    for b in benchmarks:
        path = os.path.join(log_dir, b)
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            issues.append(f"Missing benchmark: {b}")
        print(f"  {b}: [{status}]")

    # ── 3. Check round counts ────────────────────────────────────
    print("\n[3] Checking round counts...")
    all_logs = glob.glob(os.path.join(log_dir, "*.json"))
    for f in sorted(all_logs):
        if "benchmark" in os.path.basename(f) or "convergence" in os.path.basename(f):
            continue
        with open(f, "r") as fh:
            data = json.load(fh)
        rounds = data.get("rounds", [])
        basename = os.path.basename(f)
        if len(rounds) > 0:
            final_auc = rounds[-1].get("mean_auc", 0)
            status = "OK" if len(rounds) >= 30 else f"SHORT ({len(rounds)} rounds)"
            print(f"  {basename}: {len(rounds)} rounds, AUC={final_auc:.4f}  [{status}]")

    # ── 4. Comparison table ──────────────────────────────────────
    print("\n[4] Summary Table (Final Round AUC)")
    print("-" * 70)
    print(f"{'Config':<25} {'Baseline':<15} {'TeleZK-FL':<15} {'FL+DP':<15} {'Delta':<10}")
    print("-" * 70)

    for dataset, partition in configs:
        # Load each mode's logs
        baseline_aucs = extract_final_auc(
            load_logs(os.path.join(log_dir, f"baseline_{dataset}_{partition}_*.json"))
        )
        telezk_aucs = extract_final_auc(
            load_logs(os.path.join(log_dir, f"{dataset}_{partition}_*.json"))
        )
        dp_aucs = extract_final_auc(
            load_logs(os.path.join(log_dir, f"dp_{dataset}_{partition}_*.json"))
        )

        def fmt(aucs):
            if not aucs:
                return "---"
            return f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}"

        delta = ""
        if baseline_aucs and telezk_aucs:
            d = abs(np.mean(baseline_aucs) - np.mean(telezk_aucs))
            delta = f"{d:.4f}"
            if d > 0.02:
                issues.append(f"{dataset}/{partition}: AUC gap > 2% ({d:.4f})")

        print(f"  {dataset}/{partition:<20} {fmt(baseline_aucs):<15} "
              f"{fmt(telezk_aucs):<15} {fmt(dp_aucs):<15} {delta}")

    # ── 5. Check proof times ─────────────────────────────────────
    print("\n[5] Proof time variance...")
    latency_path = os.path.join(log_dir, "latency_benchmark.json")
    if os.path.exists(latency_path):
        with open(latency_path, "r") as f:
            lat_data = json.load(f)
        for entry in lat_data.get("layer_benchmarks", []):
            size = entry["layer_size"]
            std = entry["telezk_ms"]["std"]
            mean = entry["telezk_ms"]["mean"]
            cv = std / mean if mean > 0 else 0
            status = "OK" if cv > 0.01 else "SUSPICIOUS (no variance)"
            print(f"  {size}x{size}: mean={mean:.2f}ms, std={std:.2f}ms, CV={cv:.3f}  [{status}]")

    # ── 6. Scalability linearity ─────────────────────────────────
    print("\n[6] Scalability linearity...")
    scale_path = os.path.join(log_dir, "scalability_benchmark.json")
    if os.path.exists(scale_path):
        with open(scale_path, "r") as f:
            scale_data = json.load(f)
        is_linear = scale_data.get("is_linear", None)
        cv = scale_data.get("coefficient_of_variation", 0)
        print(f"  Linear: {is_linear}, CV: {cv:.3f}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if issues:
        print(f"ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    validate_results()
