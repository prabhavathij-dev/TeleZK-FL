"""
LaTeX Table Generator (Step 35).

Generates publication-ready LaTeX tables from experiment logs.
Tables can be copy-pasted directly into the .tex manuscript.

Usage: python experiments/generate_tables.py
"""
import sys
import os
import json
import glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_logs(pattern):
    """Load all JSON logs matching a glob pattern."""
    results = []
    for f in sorted(glob.glob(pattern)):
        with open(f, "r") as fh:
            results.append(json.load(fh))
    return results


def _final_auc(logs):
    """Get final-round mean AUC from each log."""
    aucs = []
    for log in logs:
        rounds = log.get("rounds", [])
        if rounds:
            aucs.append(rounds[-1].get("mean_auc", 0))
    return aucs


def _final_per_class(logs):
    """Get final-round per-class AUC from each log."""
    all_per_class = []
    for log in logs:
        rounds = log.get("rounds", [])
        if rounds:
            all_per_class.append(rounds[-1].get("per_class_auc", {}))
    return all_per_class


def _fmt(aucs, decimals=3):
    """Format mean±std."""
    if not aucs:
        return "---"
    m, s = np.mean(aucs), np.std(aucs)
    return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"


def table1_accuracy_comparison(log_dir: str):
    """TABLE 1: Diagnostic Accuracy Comparison (IID)."""
    print("\n% TABLE 1: Diagnostic Accuracy Comparison (IID)")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Comparison of Diagnostic Accuracy across Datasets (IID)}")
    print("\\label{tab:accuracy}")
    print("\\begin{tabular}{llc}")
    print("\\hline")
    print("\\textbf{Dataset} & \\textbf{Framework} & \\textbf{Mean AUC} \\\\")
    print("\\hline")

    for dataset, label in [("chexpert", "CheXpert"), ("ptbxl", "PTB-XL")]:
        baseline = _final_auc(_load_logs(os.path.join(log_dir, f"baseline_{dataset}_iid_*.json")))
        telezk = _final_auc(_load_logs(os.path.join(log_dir, f"{dataset}_iid_*.json")))
        dp = _final_auc(_load_logs(os.path.join(log_dir, f"dp_{dataset}_iid_*.json")))

        print(f"{label} & Standard FL & {_fmt(baseline)} \\\\")
        print(f" & TeleZK-FL & {_fmt(telezk)} \\\\")
        print(f" & FL+DP ($\\varepsilon$=2) & {_fmt(dp)} \\\\")
        print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")


def table2_per_pathology(log_dir: str):
    """TABLE 2: Per-Pathology AUC Breakdown on CheXpert."""
    print("\n% TABLE 2: Per-Pathology AUC on CheXpert (IID)")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Per-Pathology AUC Breakdown on CheXpert (IID)}")
    print("\\label{tab:per_pathology}")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Pathology} & \\textbf{Standard FL} & \\textbf{TeleZK-FL} \\\\")
    print("\\hline")

    baseline_pcs = _final_per_class(_load_logs(
        os.path.join(log_dir, f"baseline_chexpert_iid_*.json")))
    telezk_pcs = _final_per_class(_load_logs(
        os.path.join(log_dir, f"chexpert_iid_*.json")))

    # Collect all pathology names
    all_names = set()
    for pc in baseline_pcs + telezk_pcs:
        all_names.update(pc.keys())

    for name in sorted(all_names):
        b_vals = [pc.get(name, 0) for pc in baseline_pcs]
        t_vals = [pc.get(name, 0) for pc in telezk_pcs]
        print(f"{name} & {_fmt(b_vals)} & {_fmt(t_vals)} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def table3_noniid_impact(log_dir: str):
    """TABLE 3/5: IID vs Non-IID Accuracy."""
    print("\n% TABLE 5: Diagnostic Accuracy under IID and Non-IID")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Impact of Non-IID Data Distribution}")
    print("\\label{tab:noniid}")
    print("\\begin{tabular}{llcc}")
    print("\\hline")
    print("\\textbf{Dataset} & \\textbf{Framework} & \\textbf{IID} & \\textbf{Non-IID} \\\\")
    print("\\hline")

    for dataset, label in [("chexpert", "CheXpert"), ("ptbxl", "PTB-XL")]:
        for mode, fw in [("baseline", "Standard FL"), ("", "TeleZK-FL")]:
            prefix = f"{mode}_{dataset}" if mode else dataset
            iid = _final_auc(_load_logs(os.path.join(log_dir, f"{prefix}_iid_*.json")))
            noniid = _final_auc(_load_logs(os.path.join(log_dir, f"{prefix}_dirichlet_*.json")))
            print(f"{label} & {fw} & {_fmt(iid)} & {_fmt(noniid)} \\\\")
        print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")


def table4_latency(log_dir: str):
    """TABLE 4: Proof Generation Latency per Layer."""
    lat_path = os.path.join(log_dir, "latency_benchmark.json")
    if not os.path.exists(lat_path):
        print("\n% TABLE 4: SKIPPED (no latency data)")
        return

    with open(lat_path, "r") as f:
        data = json.load(f)

    print("\n% TABLE 4: Proof Generation Latency per Layer on RPi4")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Proof Generation Latency per Layer on RPi4}")
    print("\\label{tab:latency}")
    print("\\begin{tabular}{rccc}")
    print("\\hline")
    print("\\textbf{Layer Size} & \\textbf{Standard ZK (ms)} & "
          "\\textbf{TeleZK-FL (ms)} & \\textbf{Speedup} \\\\")
    print("\\hline")

    for entry in data.get("layer_benchmarks", []):
        sz = entry["layer_size"]
        std = entry["standard_zk_ms"]["mean"]
        tlz = entry["telezk_ms"]["mean"]
        spd = entry["speedup"]
        print(f"${sz} \\times {sz}$ & {std:.2f} & {tlz:.2f} & {spd:.1f}$\\times$ \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def table_ablation(log_dir: str):
    """TABLE 3: Ablation on Quantization Bit-Width."""
    print("\n% TABLE 3: Impact of Quantization Bit-Width")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Impact of Quantization Bit-Width (CheXpert IID)}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("\\textbf{Bit-Width} & \\textbf{Mean AUC} & "
          "\\textbf{Proof Time (ms)} & \\textbf{Comm. (MB)} \\\\")
    print("\\hline")

    for bits in [4, 8, 16, 32]:
        abl_path = os.path.join(log_dir, f"ablation_{bits}bit.json")
        if os.path.exists(abl_path):
            with open(abl_path, "r") as f:
                data = json.load(f)
            auc = data.get("final_auc", "---")
            proof_t = data.get("proof_time_ms", "---")
            comm = data.get("comm_mb", "---")
            if isinstance(auc, float):
                auc = f"{auc:.3f}"
            if isinstance(proof_t, float):
                proof_t = f"{proof_t:.2f}"
            if isinstance(comm, float):
                comm = f"{comm:.2f}"
        else:
            auc, proof_t, comm = "---", "---", "---"
        label = f"INT{bits}" if bits < 32 else "FP32"
        print(f"{label} & {auc} & {proof_t} & {comm} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_all_tables(log_dir: str = "results/logs"):
    """Generate all LaTeX tables."""
    print("%" * 70)
    print("% TeleZK-FL Paper Tables (auto-generated)")
    print("%" * 70)

    table1_accuracy_comparison(log_dir)
    table2_per_pathology(log_dir)
    table3_noniid_impact(log_dir)
    table4_latency(log_dir)
    table_ablation(log_dir)

    print("\n% End of auto-generated tables")


if __name__ == "__main__":
    generate_all_tables()
