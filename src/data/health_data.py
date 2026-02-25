import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HealthDataGenerator:
    """Generates synthetic IoMT data for health monitoring."""
    
    def __init__(self, num_samples=1000, num_features=5):
        self.num_samples = num_samples
        self.num_features = num_features
        
    def generate_data(self):
        # Features: Heart Rate, BP Systolic, BP Diastolic, SPO2, Temp
        X = np.random.randn(self.num_samples, self.num_features).astype(np.float32)
        
        # Simple rule for "Healthy" (0) vs "Needs Attention" (1)
        # Higher values generally mean more risk in this synthetic model
        y = (np.sum(X, axis=1) > 0.5).astype(np.float32)
        
        return X, y

class IoMTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(X, y, batch_size=32):
    dataset = IoMTDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
