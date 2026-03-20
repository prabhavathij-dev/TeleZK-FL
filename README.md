# TeleZK-FL: Verifiable Remote Patient Monitoring

Code for **"TeleZK-FL: Enabling Trustless and Verifiable Remote Patient 
Monitoring via Quantized Zero-Knowledge Federated Learning"** 
(Frontiers in Digital Health, 2026).

## Overview

TeleZK-FL addresses the prover bottleneck in Zero-Knowledge Federated 
Learning for telehealth via:

1. **Post-Training Quantization (PTQ)**: FP32 → INT8 gradient compression
2. **LUT-based ZK Proofs**: O(1) lookup arguments replacing O(N²·32) 
   arithmetic constraints
3. **Edge-device feasibility**: Validated under Raspberry Pi 4 
   resource constraints

## Datasets

- **CheXpert** (chest X-rays): Download from [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert)
- **PTB-XL** (12-lead ECG): Download from [PhysioNet](https://physionet.org/content/ptb-xl/)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run single experiment
python main.py --config config/chexpert_iid.yaml

# Run all 12 configurations (4 configs × 3 seeds)
python main.py --run-all

# Run benchmarks only (latency, energy, scalability)
python main.py --benchmark-only

# Generate paper figures
python main.py --generate-figures
```

## Results

| Dataset | Partition | Framework | Mean AUC |
|---------|-----------|-----------|----------|
| CheXpert | IID | Standard FL | 0.865 ± 0.003 |
| CheXpert | IID | TeleZK-FL | 0.862 ± 0.003 |
| PTB-XL | IID | Standard FL | 0.894 ± 0.004 |
| PTB-XL | IID | TeleZK-FL | 0.891 ± 0.004 |

## Citation

If you use this code, please cite:
```bibtex
@article{jayaraman2026telezk,
  title={TeleZK-FL: Enabling Trustless and Verifiable Remote Patient 
         Monitoring via Quantized Zero-Knowledge Federated Learning},
  author={Jayaraman, Prabhavathi and Delhibabu, Radhakrishnan},
  journal={Frontiers in Digital Health},
  year={2026}
}
```

## License

MIT License
