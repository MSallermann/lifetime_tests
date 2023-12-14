import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

# THIS = Path(__file__).parent
TRAJECTORY_FOLDER = Path(
    "/home/moritz/Thesis_Code/lifetime_tests/single_field/trajectories_field_max_T"
)

# TEMPERATURE_LIST = np.linspace(3.0, 6.6, 10)
TEMPERATURE_LIST = np.linspace(1.0, 3.8, 10)[:]
DAMPING_LIST = [0.3]

# TEMPERATURE_LIST = [2.0]
# DAMPING_LIST = np.linspace(0.6, 2, 40)[1:]
MIN_SAMPLES = 10

order_param = np.linspace(-1, 0.9, 100)


# Calculate the order param from one row of the trajectory file
def get_order_param(row):
    return row[-1]  # The spin z component


for temp_folder in TRAJECTORY_FOLDER.glob("./*"):
    print(f"Processing {temp_folder}")

    list_of_trajectories = list(temp_folder.glob("trajectory_*"))
    n_trajectories = len(list_of_trajectories)

    order_param_passage_times = np.zeros(shape=(n_trajectories, len(order_param)))

    for idx_traj, traj_file in enumerate(list_of_trajectories):
        print(f".... file {traj_file}")

        if traj_file.suffix == ".npy":
            data_current = np.load(traj_file)
        else:
            data_current = np.loadtxt(traj_file)

        # idx of the highest achieved order parameter so far
        idx_max_order_param = 0

        # import matplotlib.pyplot as plt
        # plt.plot( data_current[:,0], data_current[:,-1] )
        # plt.show()
        t0 = data_current[0][0]

        # TODO: express this loop in numpy somehow?
        for row in data_current[:]:
            t = row[0] - t0
            o = get_order_param(row)
            omax = order_param[idx_max_order_param]

            # Is the current order param higher than the current max?
            # If yes, we add the time to the passage times array
            if o > omax:
                # print()
                # print(t, o, omax, idx_max_order_param)
                # print(order_param_passage_times[idx_traj])

                idx_new_max = np.argmax(order_param > o)

                if idx_new_max == 0:
                    idx_new_max = len(order_param)
                    order_param_passage_times[idx_traj][
                        idx_max_order_param:idx_new_max
                    ] = np.nan
                    break

                order_param_passage_times[idx_traj][
                    idx_max_order_param:idx_new_max
                ] += t
                idx_max_order_param = idx_new_max

            if o >= np.max(order_param):
                # print(order_param_passage_times[idx_traj])
                break

    # Finally, we divide by the number of trajectories to get the mean first passage times
    n_order_param = len(order_param)

    mean_passage_times = np.zeros(n_order_param)
    std_passage_times = np.zeros(n_order_param)
    n_samples = np.zeros(n_order_param)

    for idx_o in range(n_order_param):
        times = order_param_passage_times[:, idx_o]
        n = len(times[~np.isnan(times)])
        n_samples[idx_o] = n
        if n == 0:
            continue
        mean_passage_times[idx_o] = np.mean(times[~np.isnan(times)])
        std_passage_times[idx_o] = np.std(times[~np.isnan(times)]) / np.sqrt(n)

    x = order_param[n_samples > MIN_SAMPLES]
    mean_passage_times = mean_passage_times[n_samples > MIN_SAMPLES]
    std_passage_times = std_passage_times[n_samples > MIN_SAMPLES]
    n_samples = n_samples[n_samples > MIN_SAMPLES]

    def sigmoid(x, a, b, c, f):
        y = f * (1.0 + np.exp(-b * (x - a))) ** (-1) + c
        return y

    y = mean_passage_times

    a0 = (x[-1] + x[0]) / 2
    f0 = y[-1] - y[0]
    b0 = 4 * (y[-1] - y[0]) / (x[-1] - x[0]) / f0
    c0 = -(y[-1] + y[0]) / 8
    p0 = [a0, b0, c0, f0]

    try:
        popt, pcov = curve_fit(sigmoid, x, y, p0=p0, method="lm")
        inflection_point = popt[0]  # x = a
        lifetime = sigmoid(inflection_point, *p0)
    except:
        lifetime = -1

    with open(temp_folder / "lifetime.txt", "w") as f:
        print(lifetime, file=f)

    # We save the mean passage times in the same folder we have the trajectories
    np.savetxt(
        temp_folder / "mean_times.txt",
        np.column_stack((x, mean_passage_times, std_passage_times, n_samples)),
        header="order_parameter, mean_first_passage_time, error_mean_first_passage_time, n_samples",
    )
