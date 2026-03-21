"""
Master Federated Learning Training Loop for TeleZK-FL v2.

Orchestrates the complete FL pipeline: dataset loading, partitioning,
client training, quantization, ZK proof generation, server aggregation,
evaluation, and logging.
"""

import os
import copy
import time
import math
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Literal

from src.data.chexpert_loader import get_chexpert_train_test, CHEXPERT_PATHOLOGIES
from src.data.ptbxl_loader import get_ptbxl_train_test, PTBXL_SUPERCLASSES
from src.data.partition import partition_iid, partition_dirichlet, print_partition_stats
from src.models.mobilenetv2_2d import get_mobilenetv2_2d, count_parameters
from src.models.mobilenetv2_1d import get_mobilenetv2_1d
from src.quantization.lut_builder import build_int8_multiplication_lut, load_lut
from src.zkp.prover import TeleZKProver
from src.zkp.verifier import TeleZKVerifier
from src.fl.client import FLClient
from src.fl.server import FLServer
from src.utils.rpi_simulator import RPi4Simulator
from src.utils.metrics import compute_communication_cost
from src.utils.logger import ExperimentLogger


def _set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _apply_dp_noise(
    delta: Dict[str, torch.Tensor],
    clip_norm: float = 1.0,
    epsilon: float = 2.0,
    delta_dp: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """Apply differential privacy: gradient clipping + Gaussian noise.

    Args:
        delta: Weight update dict.
        clip_norm: Maximum L2 norm for clipping (C).
        epsilon: Privacy budget.
        delta_dp: Failure probability.

    Returns:
        Noised delta dict.
    """
    noised = {}
    for name, tensor in delta.items():
        # Clip: if ||delta|| > C, scale it down
        norm = tensor.norm(2).item()
        if norm > clip_norm:
            tensor = tensor * (clip_norm / norm)

        # Add Gaussian noise: sigma = C * sqrt(2 * ln(1.25/delta)) / epsilon
        sigma = clip_norm * math.sqrt(2 * math.log(1.25 / delta_dp)) / epsilon
        noise = torch.randn_like(tensor) * sigma
        noised[name] = tensor + noise

    return noised


def run_federated_experiment(
    config_path: str,
    seed_override: Optional[int] = None,
    mode: Literal["telezk", "baseline", "dp"] = "telezk",
) -> dict:
    """Run a single federated learning experiment.

    Args:
        config_path: Path to YAML config file.
        seed_override: Override the seed from config.
        mode: Experiment mode:
            'telezk'   - Full TeleZK-FL (quantization + ZK proofs)
            'baseline' - Standard FedAvg (FP32 deltas, no quant/ZK)
            'dp'       - FedAvg + Differential Privacy (ε=2, no quant/ZK)

    Returns:
        Results dict with per-round metrics.
    """
    # 1. Load config
    config = _load_config(config_path)
    seed = seed_override or config.get("experiment", {}).get("seeds", [42])[0]
    config["_current_seed"] = seed
    config["_mode"] = mode
    _set_seed(seed)

    use_quant_zk = (mode == "telezk")
    use_dp = (mode == "dp")

    dataset_name = config["dataset"]["name"]
    partition_type = config["federated"]["partition"]
    num_clients = config["federated"]["num_clients"]
    num_rounds = config["federated"]["num_rounds"]
    local_epochs = config["federated"]["local_epochs"]

    mode_label = {"telezk": "TeleZK-FL", "baseline": "Standard FL", "dp": "FL+DP"}[mode]
    print(f"\n{'='*60}")
    print(f"{mode_label} Experiment: {dataset_name} / {partition_type} / seed={seed}")
    print(f"{'='*60}")

    # Detect device early — CUDA should be used for all training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [DEVICE] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"  [DEVICE] Using: {device}")
    if torch.cuda.is_available():
        print(f"  [DEVICE] GPU: {torch.cuda.get_device_name(0)}")

    # 2-3. Load dataset
    print("\nLoading dataset...")
    if dataset_name == "chexpert":
        data_dir = config["dataset"]["data_dir"]
        label_csv = config["dataset"]["label_csv"]
        train_dataset, test_dataset = get_chexpert_train_test(data_dir, label_csv)
        class_names = CHEXPERT_PATHOLOGIES
        num_classes = 5
    elif dataset_name == "ptbxl":
        data_dir = config["dataset"]["data_dir"]
        sampling_rate = config["dataset"].get("sampling_rate", 100)
        train_dataset, test_dataset = get_ptbxl_train_test(data_dir, sampling_rate)
        class_names = PTBXL_SUPERCLASSES
        num_classes = 5
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 4. Partition training data
    print("\nPartitioning data across clients...")
    if partition_type == "iid":
        client_datasets = partition_iid(train_dataset, num_clients, seed)
    elif partition_type == "dirichlet":
        alpha = config["federated"].get("dirichlet_alpha", 0.5)
        client_datasets = partition_dirichlet(train_dataset, num_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown partition: {partition_type}")

    print_partition_stats(client_datasets, num_classes, class_names)

    # 5. Create test loader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 6. Initialize global model
    print("\nInitializing model...")
    model_name = config["model"]["name"]
    if model_name == "mobilenetv2_2d":
        global_model = get_mobilenetv2_2d(
            num_classes=num_classes,
            pretrained=config["model"].get("pretrained", True),
        )
    elif model_name == "mobilenetv2_1d":
        global_model = get_mobilenetv2_1d(
            in_channels=config["model"].get("in_channels", 12),
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"  Model: {model_name}, Parameters: {count_parameters(global_model):,}")
    fp32_bytes, fp32_mb = compute_communication_cost(global_model.state_dict(), 32)
    int8_bytes, int8_mb = compute_communication_cost(global_model.state_dict(), 8)
    print(f"  Communication cost: FP32={fp32_mb:.2f}MB, INT8={int8_mb:.2f}MB "
          f"({fp32_mb/int8_mb:.1f}x reduction)")

    # 7. Build LUT (only needed for TeleZK mode)
    prover = None
    verifier = None
    lut_set = set()
    if use_quant_zk:
        print("\nBuilding INT8 multiplication LUT...")
        lut_path = os.path.join("data", "mul_lut_int8.npy")
        if os.path.exists(lut_path):
            print("  Loading existing LUT...")
            _, lut_set = load_lut(lut_path)
        else:
            _, lut_set = build_int8_multiplication_lut(lut_path)

        # 8. Initialize ZK components
        prover = TeleZKProver(lut_set, l_inf_bound=config["zkp"].get("l_inf_bound", 1.0))
        verifier = TeleZKVerifier(lut_set)
    else:
        print(f"\nSkipping LUT/ZK setup (mode={mode})")

    # 9. Initialize server (verifier may be None for baseline/dp modes)
    server = FLServer(global_model, verifier)

    # Move global model to GPU for training
    global_model.to(device)

    # 10. Create clients
    print(f"\nCreating {num_clients} FL clients...")
    clients = []
    for k in range(num_clients):
        client = FLClient(
            client_id=k,
            model=global_model,
            train_dataset=client_datasets[k],
            config=config,
        )
        clients.append(client)
        print(f"  Client {k}: {client.num_samples} samples")

    # 11. Initialize logger
    # NOTE: RPi4 simulator is NOT used during FL training.
    # It caused CPU affinity restrictions. Use it only for --benchmark-only.
    logger = ExperimentLogger(config)
    rpi_sim = None  # disabled during training on purpose

    # 12. Federated training loop
    print(f"\nStarting FL training for {num_rounds} rounds...")
    print(f"{'='*60}")

    for round_num in tqdm(range(num_rounds), desc="FL Rounds"):
        round_start = time.perf_counter()

        # a. Get current global weights
        global_weights = server.get_global_weights()

        # b. Client training
        client_updates = []
        client_proofs = []
        client_sizes = []
        client_scales = []
        client_zero_points = []
        proof_times = []

        for k, client in enumerate(clients):
            try:
                # Local training
                delta = client.train_local(global_weights, local_epochs)

                if use_dp:
                    # Apply DP: clip + noise
                    delta = _apply_dp_noise(delta, clip_norm=1.0, epsilon=2.0)

                if use_quant_zk:
                    # Quantize and prove
                    if rpi_sim:
                        (q_delta, proof, scales, zps), proof_time = rpi_sim.timed_run(
                            client.quantize_and_prove, delta, prover
                        )
                    else:
                        q_delta, proof, scales, zps = client.quantize_and_prove(
                            delta, prover
                        )
                        proof_time = proof.total_time
                else:
                    # Baseline / DP mode: use FP32 deltas directly
                    from src.zkp.prover import ProofResult
                    q_delta = delta  # FP32 tensors
                    proof = ProofResult(is_valid=True)
                    scales = {name: torch.tensor(1.0) for name in delta}
                    zps = {name: 0 for name in delta}
                    proof_time = 0.0

                client_updates.append(q_delta)
                client_proofs.append(proof)
                client_sizes.append(client.num_samples)
                client_scales.append(scales)
                client_zero_points.append(zps)
                proof_times.append(proof_time)

            except Exception as e:
                print(f"\n  Client {k} failed: {e}")
                continue

        # c. Server verification and aggregation
        if client_updates:
            num_valid, num_total, agg_time = server.verify_and_aggregate(
                client_updates, client_proofs, client_sizes,
                client_scales, client_zero_points,
            )
        else:
            num_valid, num_total, agg_time = 0, 0, 0

        # d. Evaluate global model
        per_class_auc, mean_auc = server.evaluate(test_loader, class_names)

        round_time = time.perf_counter() - round_start

        # e. Log round metrics
        avg_proof_time_ms = (
            np.mean(proof_times) * 1000 if proof_times else 0
        )
        metrics = {
            "mean_auc": mean_auc,
            "per_class_auc": per_class_auc,
            "avg_proof_time_ms": avg_proof_time_ms,
            "num_valid_proofs": num_valid,
            "num_total_clients": num_total,
            "aggregation_time_s": agg_time,
            "round_time_s": round_time,
        }
        logger.log_round(round_num, metrics)

        # f. Print round summary
        if round_num % 5 == 0 or round_num == num_rounds - 1:
            auc_str = " | ".join(
                f"{name}: {auc:.3f}" for name, auc in per_class_auc.items()
            )
            tqdm.write(
                f"\n  Round {round_num:3d}/{num_rounds}: "
                f"Mean AUC={mean_auc:.4f} | "
                f"Proofs={num_valid}/{num_total} | "
                f"Proof={avg_proof_time_ms:.1f}ms | "
                f"Round={round_time:.1f}s"
            )
            tqdm.write(f"    {auc_str}")

        # Save checkpoint every 10 rounds
        if config.get("experiment", {}).get("save_per_round", True):
            if (round_num + 1) % 10 == 0:
                logger.save()

    # 13. Save final log
    log_path = logger.save()

    print(f"\n{'='*60}")
    print(f"Experiment complete! Final Mean AUC: {mean_auc:.4f}")
    print(f"Log saved to: {log_path}")
    print(f"{'='*60}\n")

    return logger.results


def run_all_experiments() -> None:
    """Run all 12 experiment configurations (4 configs × 3 seeds)."""
    configs = [
        "config/chexpert_iid.yaml",
        "config/chexpert_noniid.yaml",
        "config/ptbxl_iid.yaml",
        "config/ptbxl_noniid.yaml",
    ]
    seeds = [42, 123, 456]

    total = len(configs) * len(seeds)
    print(f"\nRunning {total} experiments ({len(configs)} configs × {len(seeds)} seeds)")
    print("=" * 60)

    for i, config_path in enumerate(configs):
        for j, seed in enumerate(seeds):
            run_num = i * len(seeds) + j + 1
            print(f"\n[{run_num}/{total}] {config_path} (seed={seed})")
            try:
                run_federated_experiment(config_path, seed_override=seed)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\nAll {total} experiments complete!")
