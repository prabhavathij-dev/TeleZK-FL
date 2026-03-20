"""
PTB-XL 12-Lead ECG Dataset Loader for TeleZK-FL.

Loads the PTB-XL v1.0.3 dataset with multi-label classification
for 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP.
Uses the standard fold-based train/test splitting protocol.
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset


# 5 diagnostic superclasses
PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class PTBXLDataset(Dataset):
    """PyTorch Dataset for PTB-XL 12-lead ECG classification.

    Each sample is a 12-lead ECG recorded at the specified sampling rate,
    returned as (signal_tensor, label_tensor) where signal_tensor has shape
    (12, seq_length) and label_tensor has shape (5,).

    Args:
        records: List of (signal_array, label_array) tuples.
    """

    def __init__(self, records: List[Tuple[np.ndarray, np.ndarray]]):
        self.signals = []
        self.labels = []

        for signal, label in records:
            self.signals.append(signal)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]  # shape: (seq_length, 12)

        # Transpose to channels-first: (12, seq_length)
        signal = signal.T.copy()

        # Normalize each lead independently to zero mean, unit variance
        for ch in range(signal.shape[0]):
            mean = signal[ch].mean()
            std = signal[ch].std()
            if std > 1e-8:
                signal[ch] = (signal[ch] - mean) / std
            else:
                signal[ch] = signal[ch] - mean

        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        return signal_tensor, label_tensor


def _load_raw_signals(data_dir: str, df: pd.DataFrame, sampling_rate: int = 100):
    """Load raw ECG signals from wfdb files.

    Args:
        data_dir: Root PTB-XL directory.
        df: DataFrame with filename_lr or filename_hr column.
        sampling_rate: 100 or 500 Hz.

    Returns:
        List of numpy arrays, each of shape (seq_length, 12).
    """
    if sampling_rate == 100:
        col = "filename_lr"
    else:
        col = "filename_hr"

    signals = []
    for _, row in df.iterrows():
        fpath = os.path.join(data_dir, row[col])
        try:
            record = wfdb.rdrecord(fpath)
            signals.append(record.p_signal.astype(np.float32))
        except Exception as e:
            print(f"Warning: Could not load {fpath}: {e}")
            # Fallback: zero signal
            seq_len = 1000 if sampling_rate == 100 else 5000
            signals.append(np.zeros((seq_len, 12), dtype=np.float32))

    return signals


def _build_label_mapping(data_dir: str) -> dict:
    """Build SCP code -> superclass mapping from scp_statements.csv.

    Returns:
        Dict mapping SCP code string to superclass string.
    """
    scp_df = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)

    # Filter for diagnostic codes only
    scp_df = scp_df[scp_df["diagnostic"] == 1]

    # Build mapping: scp_code -> diagnostic_class (superclass)
    mapping = {}
    for code, row in scp_df.iterrows():
        if pd.notna(row.get("diagnostic_class")):
            mapping[code] = row["diagnostic_class"]

    return mapping


def _compute_labels(
    df: pd.DataFrame,
    scp_mapping: dict,
    superclasses: List[str] = None,
) -> List[np.ndarray]:
    """Compute multi-label binary vectors from SCP codes.

    Args:
        df: DataFrame with scp_codes column.
        scp_mapping: Mapping from SCP code to superclass.
        superclasses: Ordered list of superclass names.

    Returns:
        List of numpy arrays, each of shape (num_classes,).
    """
    superclasses = superclasses or PTBXL_SUPERCLASSES
    num_classes = len(superclasses)
    class_to_idx = {c: i for i, c in enumerate(superclasses)}

    labels = []
    for _, row in df.iterrows():
        # Parse the scp_codes dict string
        scp_codes = ast.literal_eval(row["scp_codes"])

        label_vec = np.zeros(num_classes, dtype=np.float32)
        for code, confidence in scp_codes.items():
            if code in scp_mapping:
                superclass = scp_mapping[code]
                if superclass in class_to_idx:
                    # Use confidence threshold of 0 (any mention counts)
                    if confidence > 0:
                        label_vec[class_to_idx[superclass]] = 1.0

        labels.append(label_vec)

    return labels


def get_ptbxl_train_test(
    data_dir: str,
    sampling_rate: int = 100,
    superclasses: List[str] = None,
) -> Tuple[PTBXLDataset, PTBXLDataset]:
    """Load PTB-XL train and test datasets using standard fold split.

    Standard protocol: folds 1-8 for training, fold 9 for validation,
    fold 10 for testing. We combine folds 1-9 for training and use
    fold 10 for testing as is typical in FL settings.

    Args:
        data_dir: Path to PTB-XL root directory.
        sampling_rate: 100 or 500 Hz.
        superclasses: List of superclass names to classify.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    superclasses = superclasses or PTBXL_SUPERCLASSES

    # Load metadata
    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    df = pd.read_csv(db_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.astype(str)

    # Build SCP to superclass mapping
    scp_mapping = _build_label_mapping(data_dir)

    # Standard fold split
    train_df = df[df.strat_fold <= 8].copy()
    test_df = df[df.strat_fold == 10].copy()

    print(f"PTB-XL: Loading {len(train_df)} train, {len(test_df)} test records...")

    # Load signals
    train_signals = _load_raw_signals(data_dir, train_df, sampling_rate)
    test_signals = _load_raw_signals(data_dir, test_df, sampling_rate)

    # Compute labels
    train_labels = _compute_labels(train_df, scp_mapping, superclasses)
    test_labels = _compute_labels(test_df, scp_mapping, superclasses)

    # Build datasets
    train_records = list(zip(train_signals, train_labels))
    test_records = list(zip(test_signals, test_labels))

    train_dataset = PTBXLDataset(train_records)
    test_dataset = PTBXLDataset(test_records)

    print(f"PTB-XL loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_dataset, test_dataset
