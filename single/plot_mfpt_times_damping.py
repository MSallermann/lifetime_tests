import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import my_rate


# TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_test/single/trajectories")
TRAJECTORY_FOLDER = Path(
    "/home/moritz/Thesis_Code/lifetime_tests/single/trajectories_damping"
)


# TEMPERATURE_LIST = np.linspace(3.0, 6.6, 10)
TEMPERATURE_LIST = np.linspace(1.0, 3.8, 10)[:]

DAMPING_LIST = np.linspace(0.05, 0.6, 20)

TEMPERATURE_LIST = [2.0]

# DAMPING_LIST = [0.3]

lifetime_list = []
for temperature in TEMPERATURE_LIST:
    for damping in DAMPING_LIST:
        temp_folder = (
            TRAJECTORY_FOLDER / f"damping_{damping:.3f}_temperature_{temperature:.3f}"
        )
        lifetime = np.loadtxt(temp_folder / "lifetime.txt")
        lifetime_list.append(lifetime)

plt.plot(DAMPING_LIST, 2.0 * np.array(lifetime_list), marker=".")

DAMPING_LIST = np.linspace(0.01, 0.6, 20)
plt.plot(DAMPING_LIST, 1.0 / my_rate.rate(1.0, TEMPERATURE_LIST[0], DAMPING_LIST, 1))

# plt.yscale("log")
plt.xlabel("Damping []")
plt.ylabel("Lifetime [ps]")
plt.savefig("lifetime_vs_damping")
plt.show()
