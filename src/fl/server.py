import torch
import numpy as np
from src.quantization.quantizer import Quantizer

class FLServer:
    def __init__(self, model):
        self.global_model = model
        self.quantizer = Quantizer()
        
    def aggregate(self, client_updates, client_scales):
        # aggregate updates
        new_weights = {}
        
        # init weights containers
        for name, param in self.global_model.named_parameters():
            new_weights[name] = np.zeros_like(param.data.cpu().numpy())
            
        # dequantize & sum
        num_clients = len(client_updates)
        for i in range(num_clients):
            updates = client_updates[i]
            scales = client_scales[i]
            
            for name in updates:
                dequantized = self.quantizer.dequantize(updates[name], scales[name])
                new_weights[name] += dequantized
                
        # average over clients
        for name, param in self.global_model.named_parameters():
            averaged_weights = new_weights[name] / num_clients
            param.data = torch.from_numpy(averaged_weights)
            
    def get_model(self):
        return self.global_model

    def verify_proofs(self, proofs):
        # sim proof verify
        for proof in proofs:
            if not proof.endswith("-valid"):
                return False
        return True
