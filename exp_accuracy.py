import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE, EPOCHS, NUM_SAMPLES, NUM_CLASSES = 32, 2, 500, 5
IMG_SIZE = 64 # Reduced from 224 for speed and stability

# load synthetic data to bypass the 400GB CheXpert requirement for testing
# TODO: add dataloader config for full CheXpert path in future release
X_train = torch.randn(NUM_SAMPLES, 3, IMG_SIZE, IMG_SIZE) 
y_train = torch.randint(0, 2, (NUM_SAMPLES, NUM_CLASSES)).float()
X_test = torch.randn(100, 3, IMG_SIZE, IMG_SIZE)
y_test = torch.randint(0, 2, (100, NUM_CLASSES)).float()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# simple cnn proxy model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * (IMG_SIZE // 2) * (IMG_SIZE // 2), num_classes)
        # Quantization stubs
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu(self.conv(x)))
        # For quantized tensors, reshape is safer than view
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

model = SimpleCNN(num_classes=NUM_CLASSES)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# quick train loop
print("Starting Quick Training...")
model.train()
for epoch in range(EPOCHS):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
    print(f"  Epoch {epoch+1} complete.")

# eval and quantize
def evaluate(net):
    net.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = net(inputs)
            all_preds.append(torch.sigmoid(outputs).numpy())
            all_targets.append(targets.numpy())
    return roc_auc_score(np.vstack(all_targets), np.vstack(all_preds), average="macro")

print("Evaluating FP32 Model...")
auc_fp32 = evaluate(model)

# Quantize to INT8
print("Quantizing to INT8...")
model.eval()
model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack') # Use qnnpack for CPU
# For Windows/x86, 'fbgemm' is often better, but let's try to be generic
# If qnnpack is not available, it might fallback
torch.ao.quantization.prepare(model, inplace=True)
with torch.no_grad():
    for i, (inputs, _) in enumerate(train_loader):
        if i > 5: break
        model(inputs)
model_int8 = torch.ao.quantization.convert(model, inplace=False)

print("Evaluating INT8 Model...")
auc_int8 = evaluate(model_int8)

print("\n" + "="*30)
print(f"FP32 Baseline AUC: {auc_fp32:.4f}")
print(f"TeleZK INT8 AUC:   {auc_int8:.4f}")
print(f"Accuracy Retained: {(auc_int8/auc_fp32)*100:.2f}%")
print("="*30)
