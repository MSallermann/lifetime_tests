import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import my_rate
from spirit_extras import calculation_folder
import numpy as np
import energy_barrier

# TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_test/single/trajectories")
TRAJECTORY_FOLDER = Path(
    "/home/moritz/Thesis_Code/lifetime_tests/single_field/trajectories_field_max_T"
)

TEMPERATURE_LIST = []
DAMPING_LIST = []
FIELD_LIST = []

lifetime_list = []
err_lifetime_list = []

for param_set_folder in TRAJECTORY_FOLDER.glob("*"):
    f = calculation_folder.Calculation_Folder(param_set_folder)
    TEMPERATURE_LIST.append(float(f["temperature"]))
    DAMPING_LIST.append(float(f["damping"]))
    FIELD_LIST.append(float(f["field"]))

    mean_times = np.loadtxt(param_set_folder / "mean_times.txt")
    order_param = mean_times[:, 0]

    idx_min = np.argmin(np.abs(order_param))

    lifetime = mean_times[idx_min, 1]
    std_time = mean_times[idx_min, 2]
    n_sample = mean_times[idx_min, 3]

    lifetime = np.loadtxt(param_set_folder / "lifetime.txt")
    lifetime_list.append(2 * lifetime)
    err_lifetime_list.append(2 * std_time)

lifetime_list = np.array(lifetime_list)
err_lifetime_list = np.array(err_lifetime_list)


TEMPERATURE_LIST = np.array(TEMPERATURE_LIST)
DAMPING_LIST = np.array(DAMPING_LIST)
FIELD_LIST = np.array(FIELD_LIST)

idx_sort = np.argsort(FIELD_LIST)

FIELD_LIST = FIELD_LIST[idx_sort]
lifetime_list = lifetime_list[idx_sort]
err_lifetime_list = err_lifetime_list[idx_sort]

plt.fill_between(
    FIELD_LIST,
    lifetime_list - err_lifetime_list,
    lifetime_list + err_lifetime_list,
    color="C0",
    alpha=0.2,
)

plt.plot(FIELD_LIST, lifetime_list, color="C0", marker=".")

FIELD_LIST = np.linspace(0.001, np.max(FIELD_LIST), 50)
TEMPERATURE_LIST = [energy_barrier.max_T(f) for f in FIELD_LIST]

lifetimes_computed = [
    1.0
    / my_rate.htst_rate_analytical(
        field=f, damping=DAMPING_LIST[0], T=t, diffusion=True
    )
    for f, t in zip(FIELD_LIST, TEMPERATURE_LIST)
]
plt.plot(FIELD_LIST, lifetimes_computed, color="C1", label="diffusion")

lifetimes_computed = [
    1.0
    / my_rate.htst_rate_analytical(
        field=f, damping=DAMPING_LIST[0], T=t, diffusion=False
    )
    for f, t in zip(FIELD_LIST, TEMPERATURE_LIST)
]

plt.plot(FIELD_LIST, lifetimes_computed, color="C2", label="no diffusion")


lifetimes_computed = [
    1.0
    / my_rate.htst_rate_zero_mode_analytical(
        field=f, damping=DAMPING_LIST[0], T=t, diffusion=True
    )
    for f, t in zip(FIELD_LIST, TEMPERATURE_LIST)
]

plt.plot(FIELD_LIST, lifetimes_computed, color="C3", label="zero-mode")


# plt.yscale("log")
plt.xlabel("Field [T]")
plt.ylabel("Lifetime [ps]")
plt.legend()
plt.savefig("lifetime_vs_field")
plt.show()
