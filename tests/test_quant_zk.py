"""Quick test for quantization + ZK pipeline."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.quantization.ptq import quantize_model_delta, dequantize_model_delta, compute_quantization_error
from src.quantization.lut_builder import load_lut
from src.zkp.prover import TeleZKProver
from src.zkp.verifier import TeleZKVerifier

# Quantization test
d = {"w": torch.randn(64, 32) * 0.01}
q, s, z = quantize_model_delta(d)
dq = dequantize_model_delta(q, s, z)
_, mse = compute_quantization_error(d, dq)
print(f"Quant: MSE={mse:.8f}, dtype={q['w'].dtype}")

# ZK test
_, ls = load_lut()
prover = TeleZKProver(ls)
proof = prover.generate_proof(q)
print(f"Proof: valid={proof.is_valid}, time={proof.total_time*1000:.2f}ms, ops={proof.num_operations_checked}")

verifier = TeleZKVerifier(ls)
ok, vt = verifier.verify_proof(proof, q)
print(f"Verify: valid={ok}, time={vt*1000:.2f}ms")

# Partition test
from torch.utils.data import TensorDataset
from src.data.partition import partition_iid, partition_dirichlet
ds = TensorDataset(torch.randn(500, 12, 100), torch.randint(0, 2, (500, 5)).float())
iid = partition_iid(ds, 10, 42)
diri = partition_dirichlet(ds, 10, 0.5, 42)
print(f"IID: {[len(s) for s in iid]}")
print(f"Dirichlet: {[len(s) for s in diri]}")
print("ALL OK")
