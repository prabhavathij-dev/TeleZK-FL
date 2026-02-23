# TeleZK-FL: Verifiable Remote Patient Monitoring

Simulation code for **"TeleZK-FL: Enabling Trustless and Verifiable Remote Patient Monitoring via Quantized Zero-Knowledge Federated Learning"**.

## Overview
TeleZK-FL addresses the prover bottleneck in ZK-FL via:
1. **Post-Training Quantization (PTQ)**: Compressing FP32 gradients to INT8.
2. **Look-Up Table (LUT) ZKP**: Optimized cryptographic arguments for quantized data.

This project simulates the Federated Learning process and benchmarks the performance gains in speed and energy efficiency.

## Installation

```bash
pip install -r requirements.txt
```

## Execution

Execute the following command to run the 5-round FL simulation and display ZK performance benchmarks:

```bash
python main.py
```


## Repository Structure
- `src/fl/`: Federated Learning logic (Client & Server).
- `src/quantization/`: INT8 quantization tools.
- `src/zkp/`: LUT-based ZKP performance simulator.
- `src/data/`: Health data generator for IoMT devices.
- `src/utils/`: Benchmarking and utility functions.
