import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
clients = np.array([10, 20, 50, 75, 100, 150, 200])
time_verify = (clients * 0.012) + 0.05 # linear scaling proxy for halo2

plt.figure(figsize=(8, 5))
plt.plot(clients, time_verify, 'b-d', linewidth=2)
plt.axhline(y=5.0, color='r', linestyle='--', label='Timeout Threshold (5s)')
plt.xlabel('Number of Clients ($K$)', fontweight='bold')
plt.ylabel('Verification Time (s)', fontweight='bold')
plt.title('Aggregator Verification Scalability')
plt.legend()
plt.tight_layout()
plt.savefig('scalability_graph.png', dpi=300)
print("Saved scalability_graph.png")