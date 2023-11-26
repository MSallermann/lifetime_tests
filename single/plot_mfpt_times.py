import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

# TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_test/single/trajectories")
TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_tests/single/trajectories_2")


# TEMPERATURE_LIST = np.linspace(3.0, 6.6, 10)
TEMPERATURE_LIST = np.linspace(1.0, 3.8, 10)[:]

# TEMPERATURE_LIST = [3.0]

DAMPING_LIST = [0.3]

lifetime_list = []
for temperature in TEMPERATURE_LIST:
    for damping in DAMPING_LIST:
        temp_folder = (
            TRAJECTORY_FOLDER / f"damping_{damping:.3f}_temperature_{temperature:.3f}"
        )
        lifetime = np.loadtxt(temp_folder / "lifetime.txt")
        lifetime_list.append(lifetime)

    
