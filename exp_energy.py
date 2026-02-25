POWER_WATTS = 4.8      # rpi 4 avg active power
# TODO: add power measurements for other edge devices (e.g., jetson nano)
LAYERS = 50            # approx layers in mobilenetv2
TIME_STD_MS = 2621.44  # max layer time std
TIME_TELE_MS = 104.86  # max layer time telezk

energy_std = POWER_WATTS * (TIME_STD_MS / 1000) * LAYERS
energy_tele = POWER_WATTS * (TIME_TELE_MS / 1000) * LAYERS

print(f"Standard FL Energy: {energy_std:.2f} Joules")
print(f"TeleZK-FL Energy:   {energy_tele:.2f} Joules")
print(f"Efficiency Gain:    {energy_std/energy_tele:.1f}x")