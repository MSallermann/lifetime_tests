import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from spirit_extras import calculation_folder
from spirit import constants

# THIS = Path(__file__).parent
TRAJECTORY_FOLDER = Path(
    "/home/moritz/Thesis_Code/lifetime_tests/single_field/trajectories_field_max_T_2"
)

K = 1
for temp_folder in TRAJECTORY_FOLDER.glob("./*"):
    print(f"Processing {temp_folder}")

    list_of_trajectories = list(temp_folder.glob("trajectory_*"))
    n_trajectories = len(list_of_trajectories)

    f = calculation_folder.Calculation_Folder(temp_folder)
    damping = f["damping"]
    field = f["field"]

    for idx_traj, traj_file in enumerate(list_of_trajectories[:10]):
        print(f".... file {traj_file}")

        if traj_file.suffix == ".npy":
            data_current = np.load(traj_file)
        else:
            data_current = np.loadtxt(traj_file)

        sx = constants.mu_B * field / (2 * K)
        sz = -np.sqrt(1 - sx**2)
        s_min = np.array([sx, 0, sz])

        data_new = np.zeros(shape=(data_current.shape[0], 5))
        data_new[:, :4] = data_current[:, :4]

        print(s_min)
        print(data_current[0, 1:4])
        print(np.dot(data_current[:, 1:4], s_min))
        data_new[:, 4] = -np.dot(data_current[:, 1:4], s_min)

        np.save(traj_file, data_new)
        print(traj_file)

        # data_current[]
