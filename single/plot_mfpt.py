import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_test/single/trajectories")

TEMPERATURE_LIST = np.linspace(3.0, 6.6, 10)
TEMPERATURE_LIST = [3.0]

DAMPING_LIST = [0.3]

order_param = np.linspace(-1, 1, 100)
order_param_passage_times = np.zeros(len(order_param))


# Calculate the order param from one row of the trajectory file
def get_order_param(row):
    return row[-1]  # The spin z component


for temperature in TEMPERATURE_LIST:
    for damping in DAMPING_LIST:
        temp_folder = (
            TRAJECTORY_FOLDER / f"damping_{damping:.3f}_temperature_{temperature:.3f}"
        )

        mfpt_data = np.loadtxt(temp_folder / "mean_times.txt")
        plt.plot(mfpt_data[:, 0], mfpt_data[:, 1])
        plt.xlabel("s_z")
        plt.ylabel("$\\tau$")
        plt.show()
