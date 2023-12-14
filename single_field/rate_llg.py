import numpy as np
from pathlib import Path
from spirit_extras import calculation_folder
from spirit import state, io, simulation, system, hamiltonian, constants
from spirit.parameters import llg
from spirit_extras import calculation_folder
import energy_barrier

THIS = Path(__file__).parent
folder = THIS

# General settings
INPUT = (folder / "input.cfg").as_posix()
K = 1  # meV
N_EVENTS = 300
N_SHOT = 1
DT = 1e-3
MAXITER = 2000000 * 60
ITERATIONS_LOG = 100000

# Temperature runs
TEMPERATURE_LIST = np.linspace(0.5, 2.3, 20)
FIELD_LIST = [energy_barrier.max_field(t) for t in TEMPERATURE_LIST]
DAMPING_LIST = [0.3 for _ in TEMPERATURE_LIST]
TRAJECTORY_FOLDER_NAME = "trajectories_temperature_max_field"

# Damping runs
# DAMPING_LIST = np.linspace(0.1, 1.5, 20)
# TEMPERATURE_LIST = [20.5 for _ in DAMPING_LIST]
# FIELD_LIST = [0 for _ in DAMPING_LIST]
# TRAJECTORY_FOLDER_NAME = "trajectories_damping_0T"

# Field runs
# FIELD_LIST = np.linspace(0.0, 20, 10)
# TEMPERATURE_LIST = [1.5 for _ in FIELD_LIST]
# TEMPERATURE_LIST = [energy_barrier.max_T(f) for f in FIELD_LIST]
# DAMPING_LIST = [0.3 for _ in FIELD_LIST]
# TRAJECTORY_FOLDER_NAME = "trajectories_field_max_T"

data = []

for field, temperature, damping in zip(FIELD_LIST, TEMPERATURE_LIST, DAMPING_LIST):
    sx = constants.mu_B * field / (2 * K)

    s_min = np.array([sx, 0, -np.sqrt(1 - sx**2)])

    def reset_to_initial(p_state):
        spins = system.get_spin_directions(p_state)
        spins[0] = s_min

    # Create a parent folder for all the current trajectories

    trajectory_folder = calculation_folder.Calculation_Folder(
        folder
        / TRAJECTORY_FOLDER_NAME
        / f"damping_{damping:.3f}_temperature_{temperature:.3f}_field_{field:.3f}",
        create=True,
    )

    trajectory_folder["temperature"] = temperature
    trajectory_folder["damping"] = damping
    trajectory_folder["field"] = field
    trajectory_folder.to_desc()

    life_time_list = []

    # Now find N_EVENTS trajectories
    for n_switching in range(N_EVENTS):
        trajectory_file = trajectory_folder / f"trajectory_{n_switching}.npy"
        print(trajectory_file)

        # If the file exists we skip
        if trajectory_file.exists():
            print("    Skipped")
            continue
        else:
            print("    Saving empty file")
            np.save(trajectory_file, [])

        # trajectory for this run
        trajectory = []

        with state.State(INPUT, quiet=True) as p_state:
            llg.set_timestep(p_state, DT)
            llg.set_damping(p_state, damping)
            llg.set_temperature(p_state, temperature)
            hamiltonian.set_field(p_state, magnitude=field, direction=[1, 0, 0])
            hamiltonian.set_anisotropy(p_state, magnitude=K, direction=[0, 0, 1])

            # Always start from the initial image
            reset_to_initial(p_state)

            # pointer to current spin directions
            spin_directions = system.get_spin_directions(p_state)

            simulation.start(
                p_state,
                simulation.METHOD_LLG,
                simulation.SOLVER_RK4,
                single_shot=True,
                n_iterations=MAXITER,
                n_iterations_log=ITERATIONS_LOG,
            )

            t = simulation.get_time(p_state)
            trajectory.append([0, *spin_directions[0], -1])

            while simulation.running_on_image(p_state):
                simulation.n_shot(p_state, N_SHOT)
                t = simulation.get_time(p_state)

                order_parameter = -np.dot(spin_directions[0], s_min)
                trajectory.append([t, *spin_directions[0], order_parameter])

                # If a switching event has been detected, we save the trajectory and stop the run
                if order_parameter > 0.95:
                    life_time_list.append(t)
                    print(
                        f"Found event {n_switching}/{N_EVENTS} with lifetime {t} ps ( { int((t) / DT) } steps )"
                    )
                    simulation.stop_all(p_state)

        np.save(
            trajectory_file,
            trajectory
            # header="t, sz, sy, sz",
        )
