"""
Experiment Logger for TeleZK-FL.

Logs per-round metrics to JSON files for reproducibility and
later analysis/figure generation.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentLogger:
    """Logger that saves per-round experiment metrics to JSON.

    Args:
        config: Experiment configuration dict.
        output_dir: Directory to save log files.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str = "results/logs"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Build a unique filename
        dataset = config.get("dataset", {}).get("name", "unknown")
        partition = config.get("federated", {}).get("partition", "unknown")
        seed = config.get("_current_seed", 42)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.filename = f"{dataset}_{partition}_{seed}_{timestamp}.json"
        self.filepath = os.path.join(output_dir, self.filename)

        # Initialize results structure
        self.results = {
            "config": config,
            "start_time": datetime.now().isoformat(),
            "rounds": [],
            "summary": {},
        }

    def log_round(self, round_num: int, metrics_dict: Dict[str, Any]) -> None:
        """Log metrics for a single FL round.

        Args:
            round_num: Current round number.
            metrics_dict: Dict with keys like mean_auc, per_class_auc,
                avg_proof_time_ms, num_valid_proofs, round_time_s, etc.
        """
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            **metrics_dict,
        }
        self.results["rounds"].append(entry)

    def save(self) -> str:
        """Write the complete results to JSON file.

        Returns:
            Path to the saved file.
        """
        self.results["end_time"] = datetime.now().isoformat()

        # Compute summary stats from rounds
        if self.results["rounds"]:
            aucs = [r.get("mean_auc", 0) for r in self.results["rounds"]]
            self.results["summary"]["final_auc"] = aucs[-1] if aucs else 0
            self.results["summary"]["best_auc"] = max(aucs) if aucs else 0
            self.results["summary"]["num_rounds"] = len(self.results["rounds"])

        # Convert any non-serializable types
        serializable = self._make_serializable(self.results)

        with open(self.filepath, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"Experiment log saved to: {self.filepath}")
        return self.filepath

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """Load a previously saved experiment log.

        Args:
            filepath: Path to the JSON log file.

        Returns:
            Results dict.
        """
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Recursively convert non-JSON-serializable objects."""
        if isinstance(obj, dict):
            return {k: ExperimentLogger._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ExperimentLogger._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, "item"):
            # numpy/torch scalar
            return obj.item()
        else:
            return str(obj)
