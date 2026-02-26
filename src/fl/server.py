import torch
import numpy as np
from src.quantization.quantizer import Quantizer

class FLServer:
    def __init__(self, model):
        self.global_model = model
        self.quantizer = Quantizer()
        
    def aggregate(self, client_updates, client_scales, client_zero_points):
        # aggregate updates (deltas)
        new_deltas = {}
        
        # init deltas containers
        for name, param in self.global_model.named_parameters():
            new_deltas[name] = np.zeros_like(param.data.cpu().numpy())
            
        # dequantize & sum
        num_clients = len(client_updates)
        for i in range(num_clients):
            updates = client_updates[i]
            scales = client_scales[i]
            zps = client_zero_points[i]
            
            for name in updates:
                dequantized = self.quantizer.dequantize(updates[name], scales[name], zps[name])
                new_deltas[name] += dequantized
                
        # average over clients and apply to global model
        for name, param in self.global_model.named_parameters():
            averaged_delta = new_deltas[name] / num_clients
            # apply delta update: W = W + Delta_avg
            param.data += torch.from_numpy(averaged_delta)
            
    def get_model(self):
        return self.global_model

    def verify_proofs(self, proofs):
        # sim proof verify
        for proof in proofs:
            if not proof.endswith("-valid"):
                return False
        return True
