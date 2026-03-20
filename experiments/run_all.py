"""
Master Experiment Runner for TeleZK-FL v2.

Provides a single entry point to run all experiments in sequence.
"""

from experiments.exp_latency_real import run_latency_benchmark
from experiments.exp_energy_real import run_energy_benchmark
from experiments.exp_scalability_real import run_scalability_benchmark
from experiments.exp_convergence import run_convergence_analysis
from experiments.generate_figures import generate_all_figures


def run_all_benchmarks():
    """Run all benchmark experiments."""
    print("\n" + "=" * 70)
    print("Running ALL TeleZK-FL Benchmarks")
    print("=" * 70)

    print("\n[1/4] Latency Benchmark...")
    run_latency_benchmark()

    print("\n[2/4] Energy Benchmark...")
    run_energy_benchmark()

    print("\n[3/4] Scalability Benchmark...")
    run_scalability_benchmark()

    print("\n[4/4] Convergence Analysis...")
    run_convergence_analysis()

    print("\n" + "=" * 70)
    print("All benchmarks complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
