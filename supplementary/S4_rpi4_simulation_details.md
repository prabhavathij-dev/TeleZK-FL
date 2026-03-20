# S4: RPi4 Simulation Details

## Motivation

Physical Raspberry Pi 4 devices are not always available in cloud compute
environments. We simulate RPi4 constraints on standard hardware to
produce representative performance benchmarks.

## Simulation Components

### 1. CPU Core Restriction

```python
import psutil
p = psutil.Process(os.getpid())
p.cpu_affinity([0, 1, 2, 3])  # Restrict to 4 cores
```

The RPi4 has 4 Cortex-A72 cores. We restrict the host process to use only
4 cores via OS-level CPU affinity. This ensures the process cannot
parallelize across more cores than a physical RPi4.

### 2. Memory Limit

On Linux:
```python
import resource
limit = 4 * 1024 * 1024 * 1024  # 4 GB
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
```

On Windows, memory limits are advisory. We monitor RSS usage and warn
if it exceeds 4 GB during benchmark execution.

### 3. CPU Frequency Scaling Factor

The RPi4 runs at **1500 MHz** (1.5 GHz). Host CPUs typically run at
2.5-5.0 GHz. We apply a scaling factor:

```
f_scale = f_host_max / 1500
t_rpi4_approx = t_measured × f_scale
```

For example, on a 3.5 GHz host: `f_scale = 3500/1500 = 2.33x`

This means measured times are multiplied by 2.33 to approximate RPi4
execution time.

**Auto-detection**: The scaling factor is auto-detected using
`psutil.cpu_freq().max`. If detection fails, a default of 2.3x is used.

## Validity

### Why This Is a Reasonable Approximation

1. **CPU-bound workload**: ZK proof generation is purely CPU-bound
   (hashing, LUT lookups, array operations). It does not depend on
   GPU, disk I/O, or network.

2. **Core count matters**: Restricting to 4 cores accurately represents
   the RPi4's parallelism constraint.

3. **Frequency scaling is conservative**: The Cortex-A72 has lower IPC
   than modern x86 cores, so our frequency-based scaling may slightly
   *underestimate* RPi4 execution time (conservative).

### Limitations

1. **IPC differences**: ARM Cortex-A72 has different IPC characteristics
   than x86-64 cores. The frequency ratio is only an approximation.

2. **Cache hierarchy**: RPi4 has 1 MB L2 cache vs typical 8-32 MB on
   desktop CPUs. Cache-sensitive workloads may perform worse on RPi4
   than our simulation suggests.

3. **Memory bandwidth**: LPDDR4 at 3200 MT/s on RPi4 vs DDR4/DDR5 on
   host. For memory-intensive operations, actual RPi4 may be slower.

4. **Thermal throttling**: Real RPi4 throttles under sustained load
   (above ~80°C). Our simulation does not account for this.

## Recommendation

For publication, we recommend noting that these are *simulated* RPi4
benchmarks and that validation on physical hardware is left for future
work. The simulation provides an order-of-magnitude estimate that
demonstrates feasibility of edge-device ZK proof generation.
