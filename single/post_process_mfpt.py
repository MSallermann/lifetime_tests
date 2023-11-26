import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

TRAJECTORY_FOLDER = Path("/home/moritz/Thesis_Code/lifetime_test/single/trajectories")

TEMPERATURE_LIST = np.linspace(3.0, 6.6, 10)
# TEMPERATURE_LIST = [3.0]

DAMPING_LIST = [0.3]

order_param = np.linspace(-1, 0.9, 1000)


# Calculate the order param from one row of the trajectory file
def get_order_param(row):
    return row[-1]  # The spin z component


for temperature in TEMPERATURE_LIST:
    for damping in DAMPING_LIST:
        temp_folder = (
            TRAJECTORY_FOLDER / f"damping_{damping:.3f}_temperature_{temperature:.3f}"
        )

        list_of_trajectories = list(temp_folder.glob("trajectory_*"))
        n_trajectories = len(list_of_trajectories)

        order_param_passage_times = np.zeros(shape=(n_trajectories, len(order_param)))

        print(f"Processing {n_trajectories} trajectories\n")

        for idx_traj, traj_file in enumerate(list_of_trajectories):
            # print(f"Processing {traj_file}")

            data_current = np.loadtxt(traj_file)

            # idx of the highest achieved order parameter so far
            idx_max_order_param = 0

            # import matplotlib.pyplot as plt
            # plt.plot( data_current[:,0], data_current[:,-1] )
            # plt.show()
            t0 = data_current[0][0]
            for row in data_current[:]:
                t = row[0] - t0
                o = get_order_param(row)
                omax = order_param[idx_max_order_param]

                # Is the current order param higher than the current max?
                # If yes, we add the time to the passage times array

                if o > omax:
                    # print(t, o, omax, idx_max_order_param)

                    idx_new_max = np.argmax(order_param > o)

                    if idx_new_max == 0:
                        idx_new_max = len(order_param)

                    order_param_passage_times[idx_traj][
                        idx_max_order_param:idx_new_max
                    ] += t
                    idx_max_order_param = idx_new_max

                if o >= np.max(order_param):
                    break

        # Finally, we divide by the number of trajectories to get the mean first passage times
        mean_passage_times = np.mean(order_param_passage_times, axis=0)
        std_passage_times = np.std(order_param_passage_times, axis=0) / np.sqrt(
            n_trajectories
        )

        # print(mean_passage_times.shape)
        # print(std_passage_times.shape)

        def sigmoid(x, a, b, c, f):
            y = f * (1.0 + np.exp(-b * (x - a))) ** (-1) + c
            return y

        a0 = (order_param[-1] + order_param[0]) / 2
        b0 = (
            4
            * (mean_passage_times[-1] - mean_passage_times[0])
            / (order_param[-1] - order_param[0])
        )
        c0 = (mean_passage_times[-1] + mean_passage_times[0]) / 2
        f0 = 1
        p0 = [a0, b0, c0, f0]

        # popt, pcov = curve_fit(
        #     sigmoid, order_param, mean_passage_times, p0=p0, method="lm"
        # )

        # inflection_point = popt[0]  # x = a
        # lifetime = sigmoid(inflection_point, *popt)

        # print(f" lifetime = {lifetime}")

        # with open(temp_folder / "lifetime.txt", "w") as f:
        #     print(lifetime, file=f)

        # We save teh mean passage times in the same folder we have the trajectories
        np.savetxt(
            temp_folder / "mean_times.txt",
            np.column_stack((order_param, mean_passage_times, std_passage_times)),
            header="order_parameter, mean_first_passage_time, error_mean_first_passage_time",
        )
