"""
Energy Benchmark for TeleZK-FL v2.

Estimates energy consumption for standard ZK vs TeleZK-FL proof
generation based on measured timing under RPi4 constraints and
known RPi4 power characteristics.
"""

import os
import json
import numpy as np
import torch
from typing import Dict

from src.quantization.lut_builder import build_int8_multiplication_lut, load_lut
from src.zkp.prover import TeleZKProver
from src.utils.rpi_simulator import RPi4Simulator


# RPi4 power consumption (measured values from Pi Foundation)
RPI4_IDLE_WATTS = 3.0
RPI4_SINGLE_CORE_WATTS = 6.4
RPI4_ALL_CORE_WATTS = 7.5

# Typical smartphone battery
SMARTPHONE_BATTERY_MAH = 4000
SMARTPHONE_VOLTAGE = 3.7
SMARTPHONE_ENERGY_WH = SMARTPHONE_BATTERY_MAH * SMARTPHONE_VOLTAGE / 1000
SMARTPHONE_ENERGY_J = SMARTPHONE_ENERGY_WH * 3600  # = 53,280 J


def run_energy_benchmark(
    output_dir: str = "results/logs",
) -> Dict:
    """Run the energy consumption benchmark.

    Measures proof generation time under RPi4 constraints and
    computes energy consumption E = P × t, where P is the RPi4
    all-core power draw (7.5W).

    Args:
        output_dir: Directory to save results JSON.

    Returns:
        Results dict.
    """
    print("\n" + "=" * 60)
    print("TeleZK-FL Energy Benchmark")
    print("=" * 60)

    # Setup
    lut_path = os.path.join("data", "mul_lut_int8.npy")
    if os.path.exists(lut_path):
        _, lut_set = load_lut(lut_path)
    else:
        _, lut_set = build_int8_multiplication_lut(lut_path)

    rpi_sim = RPi4Simulator()
    prover = TeleZKProver(lut_set)

    layer_sizes = [32, 64, 128, 256, 512]
    num_trials = 10
    results = {"energy_benchmarks": [], "summary": {}}

    print(f"\n{'Layer':<8} {'Std ZK (J)':<14} {'TeleZK (J)':<14} "
          f"{'Reduction':<12} {'Battery %':<12}")
    print("-" * 60)

    total_std_energy = 0
    total_telezk_energy = 0

    for size in layer_sizes:
        # Standard ZK timing
        std_times = []
        for _ in range(num_trials):
            tensor_fp32 = np.random.randn(size, size).astype(np.float32)
            from experiments.exp_latency_real import _simulate_standard_zk_proof
            _, t = rpi_sim.timed_run(_simulate_standard_zk_proof, tensor_fp32)
            std_times.append(t)

        # TeleZK-FL timing
        telezk_times = []
        for _ in range(num_trials):
            q_tensor = torch.randint(-128, 128, (size, size), dtype=torch.int8)
            dummy = {"layer": q_tensor}
            _, t = rpi_sim.timed_run(prover.generate_proof, dummy)
            telezk_times.append(t)

        std_time_mean = np.mean(std_times)
        telezk_time_mean = np.mean(telezk_times)

        # Energy = Power × Time (full CPU load)
        std_energy = RPI4_ALL_CORE_WATTS * std_time_mean
        telezk_energy = RPI4_ALL_CORE_WATTS * telezk_time_mean

        reduction = std_energy / telezk_energy if telezk_energy > 0 else 0
        battery_pct = (telezk_energy / SMARTPHONE_ENERGY_J) * 100

        total_std_energy += std_energy
        total_telezk_energy += telezk_energy

        entry = {
            "layer_size": size,
            "standard_zk_time_s": float(std_time_mean),
            "telezk_time_s": float(telezk_time_mean),
            "standard_zk_energy_j": float(std_energy),
            "telezk_energy_j": float(telezk_energy),
            "energy_reduction": float(reduction),
            "battery_drain_pct": float(battery_pct),
        }
        results["energy_benchmarks"].append(entry)

        print(f"{size}x{size:<4} {std_energy:>10.4f}    {telezk_energy:>10.6f}    "
              f"{reduction:>8.1f}x    {battery_pct:>8.6f}%")

    # Summary
    total_reduction = (
        total_std_energy / total_telezk_energy
        if total_telezk_energy > 0 else 0
    )
    results["summary"] = {
        "total_standard_energy_j": float(total_std_energy),
        "total_telezk_energy_j": float(total_telezk_energy),
        "overall_reduction": float(total_reduction),
        "rpi4_power_watts": RPI4_ALL_CORE_WATTS,
        "smartphone_battery_j": SMARTPHONE_ENERGY_J,
    }

    print(f"\nOverall energy reduction: {total_reduction:.1f}x")
    print(f"RPi4 power draw: {RPI4_ALL_CORE_WATTS}W (all cores)")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "energy_benchmark.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")

    rpi_sim.release_constraints()
    return results
