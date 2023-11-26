import numpy as np
from pathlib import Path

N_EVENTS = 60

THIS = Path(__file__).parent
INPUT = (THIS / "input.cfg").as_posix()
INITIAL = (THIS / "initial.ovf").as_posix()
TRAJ_FOLDER = "trajectories_2"

from spirit import state, io, simulation, system
from spirit.parameters import llg

TEMPERATURE_LIST = np.linspace(1.0, 3.8, 10)[:3]
TEMPERATURE_LIST = [7.0]

DAMPING_LIST = [0.3]
N_SHOT = 100
DT = 1e-2
MAXITER = 2000000 * 60
ITERATIONS_LOG = 100000

data = []

for temperature in TEMPERATURE_LIST:
    for damping in DAMPING_LIST:
        trajectory_folder = (
            THIS / TRAJ_FOLDER / f"damping_{damping:.3f}_temperature_{temperature:.3f}"
        )

        trajectory_folder.mkdir(exist_ok=True, parents=True)
        life_time_list = []

        # Now find N_EVENTS trajectories
        for n_switching in range(N_EVENTS):
            trajectory_file = trajectory_folder / f"trajectory_{n_switching}"

            # If the file exists we skip
            if trajectory_file.exists():
                continue

            # trajectory for this run
            trajectory = []

            with state.State(INPUT, quiet=False) as p_state:
                llg.set_timestep(p_state, DT)
                llg.set_damping(p_state, damping)
                llg.set_temperature(p_state, temperature)

                # Always start from the initial image
                io.image_read(p_state, INITIAL)

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

                while simulation.running_on_image(p_state):
                    simulation.n_shot(p_state, N_SHOT)
                    t = simulation.get_time(p_state)

                    trajectory.append([t, *spin_directions[0]])

                    sz = spin_directions[0][2]

                    # If a switching event has been detected, we save the trajectory and stop the run
                    if sz > 0.95:
                        life_time_list.append(t)
                        print(
                            f"Found event {n_switching}/{N_EVENTS} with lifetime {t} ps ( { int((t) / DT) } steps )"
                        )
                        np.savetxt(
                            trajectory_file,
                            trajectory,
                            header="t, sz, sy, sz",
                        )
                        simulation.stop_all(p_state)
