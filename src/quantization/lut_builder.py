"""
INT8 Multiplication Lookup Table Builder.

Builds a 256x256 = 65,536 entry lookup table for all possible INT8
multiplication results. Used by the ZK proof system to verify that
quantized multiplications are correct via O(1) membership testing.
"""

import os
import numpy as np
from typing import Tuple, Set


def build_int8_multiplication_lut(
    save_path: str = None,
) -> Tuple[np.ndarray, Set[Tuple[int, int, int]]]:
    """Build the complete INT8 multiplication lookup table.

    For all a in [-128, 127] and b in [-128, 127]:
        c = a * b (result range: -16256 to 16384)
        Store tuple (a, b, c)

    Total entries: 256 x 256 = 65,536

    Args:
        save_path: Optional path to save the LUT as .npy file.

    Returns:
        Tuple of (lut_array, lut_set) where:
            lut_array: numpy array of shape (65536, 3) with columns [a, b, c]
            lut_set: Python set of (a, b, c) tuples for O(1) lookup
    """
    print("Building INT8 multiplication LUT (65,536 entries)...")

    entries = []
    for a in range(-128, 128):
        for b in range(-128, 128):
            c = a * b
            entries.append((a, b, c))

    lut_array = np.array(entries, dtype=np.int32)
    lut_set = set(entries)

    if save_path is None:
        save_path = os.path.join("data", "mul_lut_int8.npy")

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, lut_array)
    print(f"LUT saved to {save_path} ({len(lut_set)} entries)")

    return lut_array, lut_set


def load_lut(
    path: str = None,
) -> Tuple[np.ndarray, Set[Tuple[int, int, int]]]:
    """Load a previously saved LUT from disk.

    Args:
        path: Path to the .npy file.

    Returns:
        Tuple of (lut_array, lut_set).
    """
    if path is None:
        path = os.path.join("data", "mul_lut_int8.npy")

    lut_array = np.load(path)
    lut_set = set(map(tuple, lut_array.tolist()))
    return lut_array, lut_set


def verify_operations_in_lut(
    operations: list,
    lut_set: Set[Tuple[int, int, int]],
) -> Tuple[bool, int, int]:
    """Verify that a list of multiplication operations exist in the LUT.

    Args:
        operations: List of (a, b, c) tuples from quantized forward pass.
        lut_set: The LUT set for membership testing.

    Returns:
        Tuple of (all_valid, num_checked, num_failed).
    """
    num_checked = 0
    num_failed = 0

    for op in operations:
        a, b, c = int(op[0]), int(op[1]), int(op[2])
        num_checked += 1
        if (a, b, c) not in lut_set:
            num_failed += 1

    all_valid = (num_failed == 0)
    return all_valid, num_checked, num_failed
