"""
Centralized Training Sanity Check (Step 41).

Trains each model centrally (no FL) for a few epochs to verify learning works.
Gives a centralized upper bound for AUC before adding FL complexity.

Usage: python tests/test_centralized_training.py
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mobilenetv2_2d import get_mobilenetv2_2d
from src.models.mobilenetv2_1d import get_mobilenetv2_1d
from src.data.chexpert_loader import get_chexpert_train_test
from src.data.ptbxl_loader import get_ptbxl_train_test
from src.utils.metrics import compute_auc_per_class


def run_centralized_training(model_name, dataloaders, model, epochs, lr, device):
    """Run a basic centralized training loop."""
    print(f"\n{'='*50}")
    print(f"Starting centralized training for {model_name}")
    print(f"{'='*50}")

    train_loader = dataloaders["train"]
    val_loader = dataloaders["test"]
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Training over batches
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        train_loss /= len(train_loader)
        
        # Evaluation using metrics module
        class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"] if "CheXpert" in model_name else ["NORM", "MI", "STTC", "CD", "HYP"]
        per_class_auc, mean_auc = compute_auc_per_class(model, val_loader, class_names, device=str(device))
        
        print(f"--- Epoch {epoch}/{epochs} Summary ---")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Mean AUC:   {mean_auc:.4f}")
        for c, auc_val in per_class_auc.items():
            print(f"    {c}: {auc_val:.4f}")
        
    print(f"Finished {model_name}. Final Mean AUC: {mean_auc:.4f}\n")
    return mean_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. CheXpert Sanity Check
    print("Loading CheXpert dataset...")
    # Using larger batch size since no FL client overhead
    chexpert_train, chexpert_test = get_chexpert_train_test('data/chexpert/CheXpert-v1.0-small', 'data/chexpert/train_cheXbert.csv')
    chexpert_loaders = {
        "train": DataLoader(chexpert_train, batch_size=64, shuffle=True, num_workers=4),
        "test": DataLoader(chexpert_test, batch_size=64, shuffle=False, num_workers=4)
    }
    chexpert_model = get_mobilenetv2_2d(num_classes=5, pretrained=True)
    
    # Run 3 epochs
    chexpert_auc = run_centralized_training(
        "MobileNetV2-2D (CheXpert)", 
        chexpert_loaders, 
        chexpert_model, 
        epochs=3, 
        lr=1e-4, 
        device=device
    )
    
    if chexpert_auc < 0.70:
        print("WARNING: CheXpert AUC is suspiciously low (< 0.70). Check normalization/labels.")
    else:
        print("SUCCESS: CheXpert centralized training looks healthy.")

    # 2. PTB-XL Sanity Check
    print("Loading PTB-XL dataset...")
    ptbxl_train, ptbxl_test = get_ptbxl_train_test('data/ptbxl', sampling_rate=100)
    ptbxl_loaders = {
        "train": DataLoader(ptbxl_train, batch_size=64, shuffle=True, num_workers=4),
        "test": DataLoader(ptbxl_test, batch_size=64, shuffle=False, num_workers=4)
    }
    ptbxl_model = get_mobilenetv2_1d(num_classes=5)
    
    # Run 5 epochs (no pretrained weights, needs more time)
    ptbxl_auc = run_centralized_training(
        "MobileNetV2-1D (PTB-XL)", 
        ptbxl_loaders, 
        ptbxl_model, 
        epochs=5, 
        lr=1e-4, 
        device=device
    )
    
    if ptbxl_auc < 0.60:
        print("WARNING: PTB-XL AUC is suspiciously low (< 0.60). Check normalization/labels/mapping.")
    else:
        print("SUCCESS: PTB-XL centralized training looks healthy.")
        
    print("\nSanity Check Complete!")
    print(f"Upper Bound CheXpert: {chexpert_auc:.4f}")
    print(f"Upper Bound PTB-XL:   {ptbxl_auc:.4f}")


if __name__ == "__main__":
    main()
