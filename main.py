"""
TeleZK-FL: Enabling Trustless and Verifiable Remote Patient Monitoring
via Quantized Zero-Knowledge Federated Learning

Usage:
    python main.py --config config/chexpert_iid.yaml
    python main.py --config config/chexpert_iid.yaml --mode baseline
    python main.py --config config/chexpert_iid.yaml --mode dp
    python main.py --run-all
    python main.py --benchmark-only
    python main.py --generate-figures
    python main.py --generate-tables
    python main.py --run-ablation
    python main.py --validate
    python main.py --comm-costs
"""
import argparse
from src.fl.trainer import run_federated_experiment, run_all_experiments
from experiments.exp_latency_real import run_latency_benchmark
from experiments.exp_energy_real import run_energy_benchmark
from experiments.exp_scalability_real import run_scalability_benchmark
from experiments.generate_figures import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description="TeleZK-FL v2")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--mode", choices=["telezk", "baseline", "dp"],
                        default="telezk",
                        help="Experiment mode: telezk (default), baseline, dp")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 12 TeleZK-FL experiment configurations")
    parser.add_argument("--run-baselines", action="store_true",
                        help="Run all Standard FL baseline experiments")
    parser.add_argument("--run-dp", action="store_true",
                        help="Run all FL+DP baseline experiments")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run only latency/energy/scalability benchmarks")
    parser.add_argument("--generate-figures", action="store_true",
                        help="Generate paper figures from logged results")
    parser.add_argument("--generate-tables", action="store_true",
                        help="Generate LaTeX tables from logged results")
    parser.add_argument("--run-ablation", action="store_true",
                        help="Run ablation study (INT4/8/16/32)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all experiment results")
    parser.add_argument("--comm-costs", action="store_true",
                        help="Compute communication costs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.run_all:
        run_all_experiments()
    elif args.run_baselines:
        from experiments.run_baseline import run_all_baselines
        run_all_baselines()
    elif args.run_dp:
        from experiments.run_dp_baseline import run_all_dp
        run_all_dp()
    elif args.benchmark_only:
        run_latency_benchmark()
        run_energy_benchmark()
        run_scalability_benchmark()
    elif args.generate_figures:
        generate_all_figures()
    elif args.generate_tables:
        from experiments.generate_tables import generate_all_tables
        generate_all_tables()
    elif args.run_ablation:
        from experiments.run_ablation import run_ablation
        run_ablation()
    elif args.validate:
        from experiments.validate_results import validate_results
        validate_results()
    elif args.comm_costs:
        from experiments.compute_communication import compute_communication
        compute_communication()
    elif args.config:
        run_federated_experiment(args.config, seed_override=args.seed, mode=args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
