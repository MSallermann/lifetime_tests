import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import my_rate
from spirit_extras import calculation_folder
import numpy as np

# TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_test/single/trajectories")
TRAJECTORY_FOLDER = Path(
    "/home/moritz/Thesis_Code/lifetime_tests/single_field/trajectories_temperature"
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

    idx_min = np.argmin(np.abs(order_param - 0.0))

    lifetime = mean_times[idx_min, 1]
    std_time = mean_times[idx_min, 2]
    n_sample = mean_times[idx_min, 3]

    # lifetime = np.loadtxt(temp_folder / "lifetime.txt")
    lifetime_list.append(2 * lifetime)
    err_lifetime_list.append(2 * std_time)

lifetime_list = np.array(lifetime_list)
err_lifetime_list = np.array(err_lifetime_list)


TEMPERATURE_LIST = np.array(TEMPERATURE_LIST)
DAMPING_LIST = np.array(DAMPING_LIST)
FIELD_LIST = np.array(FIELD_LIST)

idx_sort = np.argsort(TEMPERATURE_LIST)

TEMPERATURE_LIST = TEMPERATURE_LIST[idx_sort]
lifetime_list = lifetime_list[idx_sort]
err_lifetime_list = err_lifetime_list[idx_sort]

plt.fill_between(
    TEMPERATURE_LIST,
    lifetime_list - err_lifetime_list,
    lifetime_list + err_lifetime_list,
    color="C0",
    alpha=0.2,
)

plt.plot(TEMPERATURE_LIST, lifetime_list, color="C0", marker=".")

# lifetimes_computed = [
#     1.0 / my_rate.htst_rate(field=FIELD_LIST[0], damping=DAMPING_LIST[0], T=T)
#     for T in TEMPERATURE_LIST
# ]
# plt.plot(TEMPERATURE_LIST, lifetimes_computed, color="C1")


lifetimes_computed = [
    1.0
    / my_rate.htst_rate_analytical(
        field=FIELD_LIST[0], damping=DAMPING_LIST[0], T=T, diffusion=False
    )
    for T in TEMPERATURE_LIST
]
plt.plot(TEMPERATURE_LIST, lifetimes_computed, color="C1", label="no diffusion")

lifetimes_computed = [
    1.0
    / my_rate.htst_rate_analytical(
        field=FIELD_LIST[0], damping=DAMPING_LIST[0], T=T, diffusion=True
    )
    for T in TEMPERATURE_LIST
]
plt.plot(TEMPERATURE_LIST, lifetimes_computed, color="C2", label="diffusion")


lifetimes_computed = [
    1.0
    / my_rate.htst_rate_zero_mode_analytical(
        field=FIELD_LIST[0], damping=DAMPING_LIST[0], T=T, diffusion=True
    )
    for T in TEMPERATURE_LIST
]
plt.plot(TEMPERATURE_LIST, lifetimes_computed, color="C3", label="diffusion_zeromode")


plt.yscale("log")
plt.xlabel("Temperature [K]")
plt.ylabel("Lifetime [ps]")
plt.legend()
plt.savefig("lifetime_vs_temperature")
plt.show()
