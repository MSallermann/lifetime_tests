import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def sigmoid(x, a, b, c, f):
    y = f * (1.0 + np.exp(-b * (x - a))) ** (-1) + c
    return y


# TRAJECTORY_FOLDER = Path(
#     "/home/moritz/Thesis_Code/lifetime_test/single/trajectories_run_one"
# )
TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_tests/single/trajectories")


TEMPERATURE_LIST = np.linspace(3.0, 6.6, 10)
TEMPERATURE_LIST = [2]

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
        plt.fill_between(
            mfpt_data[:, 0],
            mfpt_data[:, 1] - mfpt_data[:, 2],
            mfpt_data[:, 1] + mfpt_data[:, 2],
            alpha=0.2,
        )
        plt.plot(mfpt_data[:, 0], mfpt_data[:, 1])

        x = mfpt_data[:, 0]
        y = mfpt_data[:, 1]

        a0 = (x[-1] + x[0]) / 2
        f0 = y[-1] - y[0]
        b0 = 4 * (y[-1] - y[0]) / (x[-1] - x[0]) / f0
        c0 = -(y[-1] + y[0]) / 8
        p0 = [a0, b0, c0, f0]

        popt, pcov = curve_fit(
            sigmoid, mfpt_data[:, 0], mfpt_data[:, 1], p0=p0, method="lm"
        )

        inflection_point = popt[0]  # x = a
        lifetime = sigmoid(inflection_point, *popt)
        plt.axvline(inflection_point)
        plt.axhline(lifetime)

        plt.plot(
            mfpt_data[:, 0],
            sigmoid(mfpt_data[:, 0], *p0),
            color="red",
            lw=2,
            ls="-",
        )

        plt.plot(
            mfpt_data[:, 0],
            sigmoid(mfpt_data[:, 0], *popt),
            color="black",
            lw=2,
            ls="-",
        )

        plt.xlabel("s_z")
        plt.ylabel("$\\tau$")
        plt.savefig("sigmoid.png", dpi=300)
        plt.show()
