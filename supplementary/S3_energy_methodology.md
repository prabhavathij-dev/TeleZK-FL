# S3: Energy Measurement Methodology

## Hardware Specifications

We estimate energy consumption on the **Raspberry Pi 4 Model B** (BCM2711,
Cortex-A72 at 1.5 GHz, 4 GB LPDDR4 RAM).

### Power Draw Measurements (from Raspberry Pi Foundation)

| State | Power Draw |
|-------|-----------|
| Idle | 3.0 W |
| Single-core load | 6.4 W |
| All-core load (CPU stress) | 7.5 W |
| Maximum (USB + peripherals) | 12.5 W |

## Energy Estimation Method

Since ZK proof generation is CPU-bound and uses all available cores,
we use the **all-core load** power draw of **7.5 W**.

### Formula

```
Energy (Joules) = Power (Watts) × Time (seconds)
E = P_all_core × t_measured
E = 7.5 × t
```

### Timing Measurement

- Wall-clock time measured using `time.perf_counter()` (sub-microsecond
  resolution on both Linux and Windows)
- CPU affinity restricted to 4 cores via `psutil.Process.cpu_affinity()`
- Frequency scaling factor applied: `t_rpi4 = t_host × (f_host / 1500 MHz)`
- Each benchmark repeated 10-20 trials; mean ± std reported

### Comparison with Smartphone Battery

For context, we compare energy consumption against a typical smartphone:
- Battery: 4000 mAh at 3.7 V
- Total energy: 4000 × 3.7 / 1000 = 14.8 Wh = 53,280 J

This shows that TeleZK-FL proof generation consumes a negligible fraction
of a smartphone's battery per round.

## Limitations

1. RPi4 power draw is estimated, not directly measured with a power meter
2. Frequency scaling factor is an approximation — Cortex-A72 IPC differs
   from x86/x64 host CPUs
3. Memory access patterns differ between ARM and x86 architectures
4. We do not account for GPU (RPi4 has no GPU suitable for ML inference)
