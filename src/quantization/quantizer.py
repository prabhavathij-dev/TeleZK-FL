import torch
import numpy as np

class Quantizer:
    # post-training quantizer (8-bit default)
    
    def __init__(self, bits=8):
        self.bits = bits
        self.q_min = -(2**(bits - 1))
        self.q_max = 2**(bits - 1) - 1
        
    def quantize(self, tensor):
        # quantize fp tensor to int8
        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().numpy()
        else:
            data = tensor
            
        data_min = data.min()
        data_max = data.max()
        
        # calc scale/zero-point
        # FIXME: symmetric quantization is naive, consider affine for better precision
        max_abs = max(abs(data_min), abs(data_max))
        if max_abs == 0:
            return np.zeros_like(data, dtype=np.int8), 1.0
        
        scale = max_abs / self.q_max
        
        quantized = np.round(data / scale).astype(np.int8)
        quantized = np.clip(quantized, self.q_min, self.q_max)
        
        return quantized, scale

    def dequantize(self, quantized, scale):
        # dequantize to float32
        return (quantized.astype(np.float32) * scale)

    def get_size_reduction(self, tensor):
        # theoretical size reduction factor
        return 4.0 # 32 bits / 8 bits
