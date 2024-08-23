import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.typing import NDArray
from typing import Optional, List
from hopfion_lifetimes.calculation_setup.parameters import IOModel


class Sigmoid(IOModel):
    """
    Parameters for the sigmoid of functional shape f * (1.0 + np.exp(-b * (x - a))) ** (-1) + c
    """

    a: float
    b: float
    c: float
    f: float


class MFPTLifetimeResult(IOModel):
    lifetime: float
    std_lifetime: float
    sigmoid: Sigmoid


def sigmoid(x, a, b, c, f):
    y = f * (1.0 + np.exp(-b * (x - a))) ** (-1) + c
    return y


def get_lifetime(popt, pcov):
    x_inflection = popt[0]  # x = a
    y_inflection = sigmoid(x_inflection, *popt)

    # Calculate the partial derivatives of y with respect to the parameters at the inflection point
    da = popt[3] * (
        np.exp(-popt[1] * (x_inflection - popt[0]))
        / (1.0 + np.exp(-popt[1] * (x_inflection - popt[0]))) ** 2
    )
    db = (
        popt[3]
        * (x_inflection - popt[0])
        * (
            np.exp(-popt[1] * (x_inflection - popt[0]))
            / (1.0 + np.exp(-popt[1] * (x_inflection - popt[0]))) ** 2
        )
    )
    dc = 1.0
    df = (1.0 + np.exp(-popt[1] * (x_inflection - popt[0]))) ** (-1)

    # Calculate the variance of the y-value at the inflection point
    variance_y_inflection = (
        da**2 * pcov[0, 0]
        + db**2 * pcov[1, 1]
        + dc**2 * pcov[2, 2]
        + df**2 * pcov[3, 3]
        + 2 * da * db * pcov[0, 1]
        + 2 * da * dc * pcov[0, 2]
        + 2 * da * df * pcov[0, 3]
        + 2 * db * dc * pcov[1, 2]
        + 2 * db * df * pcov[1, 3]
        + 2 * dc * df * pcov[2, 3]
    )

    # Calculate the standard deviation of the y-value at the inflection point
    std_y_inflection = np.sqrt(variance_y_inflection)

    return y_inflection, std_y_inflection


def main(mfpt_data: NDArray[np.float64], out: Path, plot_path: Optional[Path]):

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

    x_inflection = popt[0]
    lifetime, std_lifetime = get_lifetime(popt, pcov)

    if not plot_path is None:
        plt.axvline(x_inflection, color="black", ls="--")
        plt.axhline(lifetime, color="black", ls="--")

        plt.axhline(lifetime - std_lifetime, color="grey")
        plt.axhline(lifetime + std_lifetime, color="grey")

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
            plot_path,
            dpi=500,
        )

    res = MFPTLifetimeResult(
        lifetime=lifetime,
        std_lifetime=std_lifetime,
        sigmoid=Sigmoid(a=popt[0], b=popt[1], c=popt[2], f=popt[3]),
    )
    res.to_file(out)


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
        "--plot_path",
        type=Path,
        default=None,
        required=False,
        help="Path to the plot_file",
    )

    parser.add_argument(
        "-o",
        type=Path,
        required=True,
        help="Path to the plot_file",
    )

    args = parser.parse_args()

    mfpt_data = np.loadtxt(args.mfpt_data, delimiter=",")

    main(mfpt_data, out=args.o, plot_path=args.plot_path)
