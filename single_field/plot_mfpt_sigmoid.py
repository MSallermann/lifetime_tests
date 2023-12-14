import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from spirit_extras import calculation_folder


def sigmoid(x, a, b, c, f):
    y = f * (1.0 + np.exp(-b * (x - a))) ** (-1) + c
    return y


THIS = Path(__file__).parent
TRAJECTORY_FOLDER = "/home/moritz/Thesis_Code/lifetime_tests/single_field/trajectories_field_max_T/damping_0.300_temperature_0.547_field_17.778"
TRAJECTORY_FOLDER = calculation_folder.Calculation_Folder(TRAJECTORY_FOLDER)

field = float(TRAJECTORY_FOLDER["field"])
damping = float(TRAJECTORY_FOLDER["damping"])
temperature = float(TRAJECTORY_FOLDER["temperature"])


order_param = np.linspace(-1, 1, 100)
order_param_passage_times = np.zeros(len(order_param))


mfpt_data = np.loadtxt(TRAJECTORY_FOLDER / "mean_times.txt", delimiter=",")
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

# sigma = mfpt_data[:, 2]
# sigma[0] = 0.01

popt, pcov = curve_fit(
    sigmoid, mfpt_data[:, 0], mfpt_data[:, 1], p0=p0, method="lm", absolute_sigma=True
)

print(pcov)

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
file_name = (
    f"sigmoid_damping_{damping:.3f}_temperature_{temperature:.3f}_field_{field:.3f}.png"
)
print(file_name)
plt.savefig(
    file_name,
    dpi=300,
)
plt.show()
plt.close()
