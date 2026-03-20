"""
RPi4 Hardware Simulator for TeleZK-FL.

Simulates Raspberry Pi 4 resource constraints on a regular machine:
- CPU affinity limited to 4 cores
- Memory capped at 4GB
- Timing adjusted by CPU frequency scaling factor

Since we don't have physical RPi4 hardware, this allows us to produce
representative benchmarks under simulated constraints.
"""

import os
import sys
import time
import psutil
from typing import Any, Callable, Tuple, Optional


class RPi4Simulator:
    """Simulates Raspberry Pi 4 hardware constraints.

    RPi4 specs: Cortex-A72 at 1.5GHz, 4 cores, 4GB LPDDR4 RAM.
    We simulate these by restricting CPU affinity and scaling timing.

    Args:
        num_cores: Number of CPU cores to restrict to (default: 4).
        memory_limit_gb: Soft memory limit in GB (default: 4).
        freq_scale_factor: CPU frequency scaling factor. If None,
            auto-detects from current CPU frequency vs RPi4's 1.5GHz.
    """

    RPI4_FREQ_MHZ = 1500  # Cortex-A72 at 1.5GHz

    def __init__(
        self,
        num_cores: int = 4,
        memory_limit_gb: int = 4,
        freq_scale_factor: Optional[float] = None,
    ):
        self.num_cores = num_cores
        self.memory_limit_gb = memory_limit_gb
        self.constraints_applied = False

        if freq_scale_factor is not None:
            self.freq_scale_factor = freq_scale_factor
        else:
            self.freq_scale_factor = self.detect_freq_scale()

        print(f"RPi4 Simulator initialized:")
        print(f"  Cores: {self.num_cores}")
        print(f"  Memory limit: {self.memory_limit_gb}GB")
        print(f"  Frequency scale factor: {self.freq_scale_factor:.2f}x")

    def detect_freq_scale(self) -> float:
        """Detect CPU frequency scaling factor.

        Compares host CPU frequency to RPi4's 1.5GHz Cortex-A72.

        Returns:
            Scaling factor (host_freq / rpi4_freq).
        """
        try:
            freq = psutil.cpu_freq()
            if freq and freq.max > 0:
                scale = freq.max / self.RPI4_FREQ_MHZ
                print(f"  Detected host CPU freq: {freq.max:.0f} MHz")
                return max(scale, 1.0)
        except Exception:
            pass

        # Default fallback
        print("  Could not detect CPU frequency, using default 2.3x")
        return 2.3

    def apply_constraints(self) -> None:
        """Apply CPU affinity and memory constraints.

        On Windows, uses psutil for CPU affinity.
        Memory limits are advisory only on Windows.
        """
        if self.constraints_applied:
            return

        try:
            p = psutil.Process(os.getpid())

            # Set CPU affinity to first N cores
            available_cpus = list(range(psutil.cpu_count()))
            target_cpus = available_cpus[:self.num_cores]
            p.cpu_affinity(target_cpus)
            print(f"  CPU affinity set to cores {target_cpus}")

        except Exception as e:
            print(f"  Warning: Could not set CPU affinity: {e}")
            print(f"  Falling back to frequency scaling only")

        # On Unix systems, we could set RLIMIT_AS, but on Windows
        # we rely on the frequency scaling factor for timing accuracy
        if sys.platform != "win32":
            try:
                import resource
                limit_bytes = self.memory_limit_gb * 1024 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS,
                                   (limit_bytes, limit_bytes))
                print(f"  Memory limit set to {self.memory_limit_gb}GB")
            except Exception as e:
                print(f"  Warning: Could not set memory limit: {e}")

        self.constraints_applied = True

    def timed_run(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, float]:
        """Run a function under RPi4 constraints and measure time.

        Timing is measured using perf_counter and adjusted by the
        frequency scaling factor to approximate RPi4 execution time.

        Args:
            func: Function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Tuple of (func_result, adjusted_time_seconds).
        """
        self.apply_constraints()

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Scale timing to approximate RPi4 performance
        adjusted_time = elapsed * self.freq_scale_factor

        return result, adjusted_time

    def release_constraints(self) -> None:
        """Release CPU affinity constraints (restore all cores)."""
        try:
            p = psutil.Process(os.getpid())
            all_cpus = list(range(psutil.cpu_count()))
            p.cpu_affinity(all_cpus)
            self.constraints_applied = False
            print("  CPU constraints released")
        except Exception:
            pass
