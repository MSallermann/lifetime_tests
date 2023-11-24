import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# plt.hist(life_times, 30, density=True)
# plt.show()

data = np.loadtxt(Path(__file__).parent / "data.txt")

# plt.axhline(2*0.0752640745962459, ls = "--", color="black", label = "HTST")
plt.errorbar(
    data[:, 1],
    1 / data[:, 2],
    yerr=data[:, 3] / data[:, 2] ** 2,
    marker="o",
    label="dynamics",
)
plt.ylabel("Rate [1/ps]")
plt.ylim(bottom=0)
plt.xlabel(r"Temperature [K]")
plt.legend()
plt.show()
