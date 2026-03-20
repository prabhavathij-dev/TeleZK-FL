"""
Post-Training Quantization (PTQ) for Model Weight Deltas.

Implements symmetric INT8 quantization with per-channel or global
scale factors. Used to compress FL model updates before ZK proof
generation.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int = 8,
    per_channel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize a floating-point tensor to INT8 using symmetric quantization.

    Symmetric: scale = max(|tensor|) / 127, zero_point = 0
    q = clamp(round(tensor / scale), -128, 127)

    Args:
        tensor: Float tensor to quantize.
        num_bits: Number of quantization bits (default: 8).
        per_channel: If True, compute scale per output channel (dim=0).

    Returns:
        Tuple of (quantized_tensor as int8, scale, zero_point).
    """
    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1
    zero_point = 0

    if per_channel and tensor.dim() >= 2:
        # Per-channel: compute scale along dim=0
        num_channels = tensor.shape[0]
        scale = torch.zeros(num_channels, dtype=torch.float32)

        for ch in range(num_channels):
            ch_max = tensor[ch].abs().max().item()
            if ch_max < 1e-12:
                scale[ch] = 1.0  # avoid division by zero
            else:
                scale[ch] = ch_max / q_max
    else:
        # Global scale
        tensor_max = tensor.abs().max().item()
        if tensor_max < 1e-12:
            scale = torch.tensor(1.0, dtype=torch.float32)
        else:
            scale = torch.tensor(tensor_max / q_max, dtype=torch.float32)

    # Quantize
    if per_channel and tensor.dim() >= 2:
        # Reshape scale for broadcasting
        shape = [1] * tensor.dim()
        shape[0] = -1
        scale_view = scale.view(shape)
        q = torch.clamp(torch.round(tensor / scale_view), q_min, q_max)
    else:
        q = torch.clamp(torch.round(tensor / scale), q_min, q_max)

    quantized = q.to(torch.int8)
    return quantized, scale, zero_point


def quantize_model_delta(
    delta_dict: Dict[str, torch.Tensor],
    num_bits: int = 8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int]]:
    """Quantize all layers of a model weight delta to INT8.

    Args:
        delta_dict: Dict of {layer_name: float_tensor} weight updates.
        num_bits: Quantization bit width.

    Returns:
        Tuple of (quantized_dict, scales_dict, zero_points_dict).
    """
    quantized_dict = {}
    scales_dict = {}
    zero_points_dict = {}

    for name, tensor in delta_dict.items():
        # Use per-channel for weight tensors (2D+), global for biases (1D)
        per_channel = tensor.dim() >= 2
        q, scale, zp = quantize_tensor(tensor, num_bits, per_channel)
        quantized_dict[name] = q
        scales_dict[name] = scale
        zero_points_dict[name] = zp

    return quantized_dict, scales_dict, zero_points_dict


def dequantize_model_delta(
    quantized_dict: Dict[str, torch.Tensor],
    scales_dict: Dict[str, torch.Tensor],
    zero_points_dict: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """Dequantize INT8 model deltas back to FP32.

    Reverse: tensor = (quantized.float() - zero_point) * scale

    Args:
        quantized_dict: Dict of INT8 tensors.
        scales_dict: Dict of scale factors.
        zero_points_dict: Dict of zero points.

    Returns:
        Dict of float32 tensors.
    """
    dequantized = {}

    for name, q_tensor in quantized_dict.items():
        scale = scales_dict[name]
        zp = zero_points_dict[name]

        q_float = q_tensor.float() - zp

        if isinstance(scale, torch.Tensor) and scale.dim() > 0 and q_float.dim() >= 2:
            # Per-channel dequantization
            shape = [1] * q_float.dim()
            shape[0] = -1
            scale_view = scale.view(shape)
            dequantized[name] = q_float * scale_view
        else:
            dequantized[name] = q_float * scale

    return dequantized


def compute_quantization_error(
    original_dict: Dict[str, torch.Tensor],
    dequantized_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, float], float]:
    """Compute MSE between original FP32 and dequantized deltas.

    Args:
        original_dict: Original FP32 weight deltas.
        dequantized_dict: Dequantized FP32 weight deltas.

    Returns:
        Tuple of (per_layer_mse dict, overall_mean_mse).
    """
    per_layer_mse = {}
    total_mse = 0.0
    total_elements = 0

    for name in original_dict:
        if name in dequantized_dict:
            diff = original_dict[name] - dequantized_dict[name]
            mse = (diff ** 2).mean().item()
            per_layer_mse[name] = mse
            total_mse += (diff ** 2).sum().item()
            total_elements += diff.numel()

    overall_mse = total_mse / max(total_elements, 1)
    return per_layer_mse, overall_mse
