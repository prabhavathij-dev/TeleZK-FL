"""
Smoke test: verify all TeleZK-FL v2 components load without errors.
Run: python tests/test_data_loading.py

Tests 1-2 require datasets on disk. Tests 3-6 run regardless.
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================
# TEST 1: CheXpert Dataset Loading
# ======================================================
print("=" * 60)
print("TEST 1: CheXpert Dataset Loading")
print("=" * 60)

chexpert_ok = False
data_dir = "data/chexpert/CheXpert-v1.0-small"
label_csv = "data/chexpert/train_cheXbert.csv"

if not os.path.exists(data_dir):
    print(f"  SKIP: Dataset directory not found: {data_dir}")
    print(f"  Download CheXpert from Kaggle and place in data/chexpert/")
elif not os.path.exists(label_csv):
    print(f"  SKIP: Label CSV not found: {label_csv}")
else:
    from src.data.chexpert_loader import get_chexpert_train_test
    try:
        train_ds, test_ds = get_chexpert_train_test(
            data_dir=data_dir,
            label_csv=label_csv,
        )
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Test samples:  {len(test_ds)}")

        img, label = train_ds[0]
        print(f"  Image shape:   {img.shape}")    # Expected: (3, 224, 224)
        print(f"  Label shape:   {label.shape}")   # Expected: (5,)
        print(f"  Label values:  {label.tolist()}")
        print(f"  Image dtype:   {img.dtype}")
        print("  OK: CheXpert loading passed")
        chexpert_ok = True
    except Exception as e:
        print(f"  FAIL: CheXpert: {e}")
        import traceback; traceback.print_exc()

# ======================================================
# TEST 2: PTB-XL Dataset Loading
# ======================================================
print("\n" + "=" * 60)
print("TEST 2: PTB-XL Dataset Loading")
print("=" * 60)

ptbxl_ok = False
ptbxl_dir = "data/ptbxl"
ptbxl_csv = os.path.join(ptbxl_dir, "ptbxl_database.csv")

if not os.path.exists(ptbxl_dir):
    print(f"  SKIP: Dataset directory not found: {ptbxl_dir}")
    print(f"  Download PTB-XL from PhysioNet and place in data/ptbxl/")
elif not os.path.exists(ptbxl_csv):
    print(f"  SKIP: Database CSV not found: {ptbxl_csv}")
else:
    from src.data.ptbxl_loader import get_ptbxl_train_test
    try:
        train_ds, test_ds = get_ptbxl_train_test(
            data_dir=ptbxl_dir,
            sampling_rate=100,
        )
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Test samples:  {len(test_ds)}")

        signal, label = train_ds[0]
        print(f"  Signal shape:  {signal.shape}")   # Expected: (12, 1000)
        print(f"  Label shape:   {label.shape}")    # Expected: (5,)
        print(f"  Label values:  {label.tolist()}")
        print(f"  Signal range:  [{signal.min():.3f}, {signal.max():.3f}]")
        print("  OK: PTB-XL loading passed")
        ptbxl_ok = True
    except Exception as e:
        print(f"  FAIL: PTB-XL: {e}")
        import traceback; traceback.print_exc()

# ======================================================
# TEST 3: Data Partitioning (uses dummy dataset if real not available)
# ======================================================
print("\n" + "=" * 60)
print("TEST 3: Data Partitioning")
print("=" * 60)

from src.data.partition import partition_iid, partition_dirichlet
import torch
from torch.utils.data import TensorDataset

try:
    # Use a real loaded dataset or create a dummy one
    if ptbxl_ok:
        test_part_ds = train_ds
        print("  Using loaded PTB-XL dataset for partition test")
    else:
        # Create a dummy multi-label dataset
        dummy_x = torch.randn(500, 12, 100)
        dummy_y = torch.randint(0, 2, (500, 5)).float()
        test_part_ds = TensorDataset(dummy_x, dummy_y)
        print("  Using dummy dataset (500 samples, 5 classes) for partition test")

    # Test IID
    iid_splits = partition_iid(test_part_ds, num_clients=10, seed=42)
    print(f"  IID: {len(iid_splits)} clients")
    print(f"  Samples per client: {[len(s) for s in iid_splits]}")

    # Test Dirichlet
    dir_splits = partition_dirichlet(test_part_ds, num_clients=10, alpha=0.5, seed=42)
    print(f"  Dirichlet: {len(dir_splits)} clients")
    print(f"  Samples per client: {[len(s) for s in dir_splits]}")
    print("  OK: Partitioning passed")
except Exception as e:
    print(f"  FAIL: Partitioning: {e}")
    import traceback; traceback.print_exc()

# ======================================================
# TEST 4: Model Forward Pass
# ======================================================
print("\n" + "=" * 60)
print("TEST 4: Model Forward Pass")
print("=" * 60)

from src.models.mobilenetv2_2d import get_mobilenetv2_2d, count_parameters
from src.models.mobilenetv2_1d import get_mobilenetv2_1d

try:
    # 2D model
    model_2d = get_mobilenetv2_2d(num_classes=5, pretrained=False)
    dummy_img = torch.randn(2, 3, 224, 224)
    out_2d = model_2d(dummy_img)
    params_2d = count_parameters(model_2d)
    print(f"  MobileNetV2-2D: {params_2d:,} params, output={out_2d.shape}")
    assert out_2d.shape == (2, 5), f"Expected (2,5), got {out_2d.shape}"

    # 1D model
    model_1d = get_mobilenetv2_1d(in_channels=12, num_classes=5)
    dummy_ecg = torch.randn(2, 12, 1000)
    out_1d = model_1d(dummy_ecg)
    params_1d = count_parameters(model_1d)
    print(f"  MobileNetV2-1D: {params_1d:,} params, output={out_1d.shape}")
    assert out_1d.shape == (2, 5), f"Expected (2,5), got {out_1d.shape}"

    # Verify output is in [0, 1] (sigmoid)
    assert out_2d.min() >= 0 and out_2d.max() <= 1, "2D output not in [0,1]"
    assert out_1d.min() >= 0 and out_1d.max() <= 1, "1D output not in [0,1]"
    print("  OK: Models passed")
except Exception as e:
    print(f"  FAIL: Models: {e}")
    import traceback; traceback.print_exc()

# ======================================================
# TEST 5: Quantization + LUT
# ======================================================
print("\n" + "=" * 60)
print("TEST 5: Quantization + LUT")
print("=" * 60)

from src.quantization.ptq import quantize_model_delta, dequantize_model_delta, compute_quantization_error
from src.quantization.lut_builder import build_int8_multiplication_lut

try:
    lut_array, lut_set = build_int8_multiplication_lut()
    print(f"  LUT entries: {len(lut_set)}")
    assert len(lut_set) == 65536, f"Expected 65536, got {len(lut_set)}"

    # Test quantization round-trip
    dummy_delta = {"layer1.weight": torch.randn(64, 32) * 0.01}
    q_delta, scales, zps = quantize_model_delta(dummy_delta)
    deq_delta = dequantize_model_delta(q_delta, scales, zps)

    per_layer_mse, overall_mse = compute_quantization_error(dummy_delta, deq_delta)
    print(f"  Quantization round-trip MSE: {overall_mse:.8f}")
    assert overall_mse < 0.001, f"MSE too high: {overall_mse}"

    # Verify INT8 range
    q_vals = q_delta["layer1.weight"]
    assert q_vals.dtype == torch.int8, f"Expected int8, got {q_vals.dtype}"
    print("  OK: Quantization passed")
except Exception as e:
    print(f"  FAIL: Quantization: {e}")
    import traceback; traceback.print_exc()

# ======================================================
# TEST 6: ZK Prover + Verifier
# ======================================================
print("\n" + "=" * 60)
print("TEST 6: ZK Proof Generation + Verification")
print("=" * 60)

from src.zkp.prover import TeleZKProver
from src.zkp.verifier import TeleZKVerifier

try:
    prover = TeleZKProver(lut_set=lut_set, l_inf_bound=1.0)
    proof = prover.generate_proof(q_delta)
    print(f"  Proof valid:    {proof.is_valid}")
    print(f"  Proof time:     {proof.total_time*1000:.2f} ms")
    print(f"  Ops checked:    {proof.num_operations_checked}")
    print(f"  Range violations: {proof.num_range_violations}")
    assert proof.is_valid, "Proof should be valid"

    # Verify it
    verifier = TeleZKVerifier(lut_set=lut_set)
    is_valid, v_time = verifier.verify_proof(proof, q_delta)
    print(f"  Verification:   valid={is_valid}, time={v_time*1000:.2f} ms")
    assert is_valid, "Verification should pass"

    # Layer benchmark
    bench = prover.benchmark_layer(64, num_trials=5)
    print(f"  Benchmark 64x64: {bench['mean']*1000:.2f} ms (mean)")
    print("  OK: ZK Prover + Verifier passed")
except Exception as e:
    print(f"  FAIL: ZK: {e}")
    import traceback; traceback.print_exc()

# ======================================================
# SUMMARY
# ======================================================
print("\n" + "=" * 60)
print("SMOKE TEST SUMMARY")
print("=" * 60)
tests = {
    "CheXpert Loading": chexpert_ok if "chexpert_ok" in dir() else "SKIPPED",
    "PTB-XL Loading": ptbxl_ok if "ptbxl_ok" in dir() else "SKIPPED",
    "Data Partitioning": "Check output above",
    "Models": "Check output above",
    "Quantization + LUT": "Check output above",
    "ZK Prover + Verifier": "Check output above",
}
for name, status in tests.items():
    if status is True:
        print(f"  [PASS] {name}")
    elif status is False:
        print(f"  [FAIL] {name}")
    else:
        print(f"  [----] {name}: {status}")

if not chexpert_ok:
    print("\n  NOTE: CheXpert dataset not found. Download from:")
    print("  https://www.kaggle.com/datasets/ashery/chexpert")
if not ptbxl_ok:
    print("\n  NOTE: PTB-XL dataset not found. Download from:")
    print("  https://physionet.org/content/ptb-xl/")
print("=" * 60)
