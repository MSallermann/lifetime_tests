import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.typing import NDArray


def sigmoid(x, a, b, c, f):
    y = f * (1.0 + np.exp(-b * (x - a))) ** (-1) + c
    return y


def main(mfpt_data: NDArray[np.float64], out: Path):

    x = mfpt_data[:, 0]
    y = mfpt_data[:, 1]

    a0 = (x[-1] + x[0]) / 2
    f0 = y[-1] - y[0]
    b0 = 4 * (y[-1] - y[0]) / (x[-1] - x[0]) / f0
    c0 = -(y[-1] + y[0]) / 8
    p0 = [a0, b0, c0, f0]

    popt, pcov = curve_fit(
        sigmoid,
        mfpt_data[:, 0],
        mfpt_data[:, 1],
        p0=p0,
        method="lm",
        absolute_sigma=True,
    )

    inflection_point = popt[0]  # x = a
    lifetime = sigmoid(inflection_point, *popt)
    plt.axvline(inflection_point, color="black", ls="--")
    plt.axhline(lifetime, color="black", ls="--")

    plt.fill_between(
        mfpt_data[:, 0],
        mfpt_data[:, 1] - mfpt_data[:, 2],
        mfpt_data[:, 1] + mfpt_data[:, 2],
        alpha=0.2,
        color="C0",
    )

    plt.plot(mfpt_data[:, 0], mfpt_data[:, 1], lw=2, color="C0")

    plt.plot(
        mfpt_data[:, 0],
        sigmoid(mfpt_data[:, 0], *popt),
        color="black",
        lw=2,
        ls="-",
    )

    plt.xlabel("s_z")
    plt.ylabel(r"$\tau$")

    plt.savefig(
        out,
        dpi=500,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some data.")

    parser.add_argument(
        "--mfpt_data",
        required=True,
        type=Path,
        help="Path to mfpt_data file",
    )

    parser.add_argument(
        "-o",
        type=Path,
        required=True,
        help="Path to the plot_file",
    )

    args = parser.parse_args()

    mfpt_data = np.loadtxt(args.mfpt_data, delimiter=",")

    main(mfpt_data, args.o)
