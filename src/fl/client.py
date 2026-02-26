import torch
import torch.nn as nn
import torch.optim as optim
from src.quantization.quantizer import Quantizer
import time

class SimpleModel(nn.Module):
    def __init__(self, input_dim=5):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.fc(x))

class FLClient:
    def __init__(self, client_id, dataloader, model=None):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = model if model else SimpleModel()
        self.quantizer = Quantizer()
        
    def train(self, epochs=1):
        self.global_weights = {name: param.data.clone() for name, param in self.model.named_parameters()}
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        start_time = time.time()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        training_time = time.time() - start_time
        return training_time

    def get_updates(self):
        # quantize model weights for transmission
        updates = {}
        scales = {}
        zero_points = {}
        
        for name, param in self.model.named_parameters():
            delta_w = param.data - self.global_weights[name]
            if isinstance(delta_w, torch.Tensor):
                delta_w = delta_w.cpu().numpy()
            q_data, scale, zero_point = self.quantizer.quantize_layer(delta_w)
            updates[name] = q_data
            scales[name] = scale
            zero_points[name] = zero_point
            
        return updates, scales, zero_points

    def generate_proof(self):
        # simulate zkp generation (placeholder)
        # TODO: integrate actual prover from src.zkp
        return f"zk-proof-{self.client_id}-valid"
