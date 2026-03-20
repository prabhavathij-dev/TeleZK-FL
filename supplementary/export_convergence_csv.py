"""
Export convergence logs to CSV (Step 38, S1).

Reads all experiment JSON logs and exports per-round AUC data
as a single CSV for supplementary material.

Usage: python supplementary/export_convergence_csv.py
"""
import sys
import os
import csv
import json
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def export_convergence_csv(
    log_dir: str = "results/logs",
    output_path: str = "supplementary/S1_convergence_logs.csv",
):
    """Export per-round AUC from all experiment logs to CSV."""
    print("Exporting convergence data to CSV...")

    # Find all experiment logs (exclude benchmark files)
    all_files = glob.glob(os.path.join(log_dir, "*.json"))
    experiment_files = [
        f for f in all_files
        if not any(x in os.path.basename(f)
                   for x in ["benchmark", "convergence", "ablation", "communication", "comm"])
    ]

    if not experiment_files:
        print(f"  No experiment logs found in {log_dir}")
        return

    print(f"  Found {len(experiment_files)} experiment log files")

    rows = []
    for filepath in sorted(experiment_files):
        basename = os.path.basename(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        dataset = config.get("dataset", {}).get("name", "unknown")
        partition = config.get("federated", {}).get("partition", "unknown")
        seed = config.get("_current_seed", "unknown")
        mode = config.get("_mode", "telezk")

        for round_data in data.get("rounds", []):
            row = {
                "experiment": basename,
                "dataset": dataset,
                "partition": partition,
                "mode": mode,
                "seed": seed,
                "round": round_data.get("round", 0),
                "mean_auc": round_data.get("mean_auc", 0),
            }
            # Add per-class AUC
            per_class = round_data.get("per_class_auc", {})
            for cls_name, cls_auc in per_class.items():
                row[f"auc_{cls_name}"] = cls_auc

            rows.append(row)

    if not rows:
        print("  No round data found")
        return

    # Get all column names
    all_cols = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in all_cols:
                all_cols.append(k)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Exported {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    export_convergence_csv()
