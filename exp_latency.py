import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

LAYER_SIZES = np.array([32, 64, 128, 256, 512])
OPS_PER_MS = 5000  # generic mobile cpu constraints/ms

std_times = (LAYER_SIZES**2 * 50) / OPS_PER_MS  # 50 constraints per fp32 op
telezk_times = (LAYER_SIZES**2 * 2) / OPS_PER_MS # 2 constraints per int8 lut op

plt.figure(figsize=(8, 5))
plt.plot(LAYER_SIZES, std_times, 'r--o', label='Standard ZK-FL (FP32)')
plt.plot(LAYER_SIZES, telezk_times, 'g-s', label='TeleZK-FL (INT8 + LUT)')
plt.yscale('log')
plt.xlabel('Neural Network Layer Size (N x N)', fontweight='bold')
plt.ylabel('Proof Gen. Time (ms) [Log]', fontweight='bold')
plt.title('Proof Generation Overhead on Edge Devices')
plt.legend()
plt.tight_layout()
plt.savefig('latency_graph.png', dpi=300)
print("Saved latency_graph.png")