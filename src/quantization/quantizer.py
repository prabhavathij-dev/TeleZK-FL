import torch
import numpy as np

class Quantizer:
    # post-training quantizer (8-bit default)
    
    def __init__(self, bits=8):
        self.bits = bits
        self.q_min = -(2**(bits - 1))
        self.q_max = 2**(bits - 1) - 1
        
    def quantize_layer(self, weights):
        w_min, w_max = np.min(weights), np.max(weights)
        if w_min == w_max:
            return np.zeros_like(weights, dtype=np.int8), 1.0, 0
        
        scale = (w_max - w_min) / (127 - (-128))
        zero_point = np.round((0 - w_min) / scale) + (-128)
        zero_point = np.clip(zero_point, -128, 127)
        
        q_weights = np.round(weights / scale) + zero_point
        q_weights = np.clip(q_weights, -128, 127).astype(np.int8)
        
        return q_weights, scale, int(zero_point)

    def dequantize(self, quantized, scale, zero_point):
        # dequantize using affine formula: (q - zp) * scale
        return (quantized.astype(np.float32) - zero_point) * scale

    def get_size_reduction(self, tensor):
        # theoretical size reduction factor
        return 4.0 # 32 bits / 8 bits
