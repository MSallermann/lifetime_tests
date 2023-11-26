import numpy as np
from pathlib import Path

N_EVENTS = 60

THIS = Path(__file__).parent
INPUT = (THIS / "input.cfg").as_posix()
INITIAL = (THIS / "initial.ovf").as_posix()
DATA = (THIS / "data.txt").as_posix()

from spirit import state, io, simulation, system
from spirit.parameters import llg

TEMPERATURE_LIST = np.linspace(1.0, 3.8, 10)[:3]
# TEMPERATURE_LIST = [3.0]

DAMPING_LIST = [0.3]
N_SHOT = 100
DT = 1e-2

data = []

for temperature in TEMPERATURE_LIST:
    for damping in DAMPING_LIST:
        with state.State(INPUT, quiet=False) as p_state:
            trajectory_folder = (
                THIS
                / "trajectories"
                / f"damping_{damping:.3f}_temperature_{temperature:.3f}"
            )
            trajectory_folder.mkdir(exist_ok=True, parents=True)

            llg.set_timestep(p_state, DT)
            llg.set_damping(p_state, damping)
            llg.set_temperature(p_state, temperature)

            # trajectory for this run
            trajectory = []

            io.image_read(p_state, INITIAL)

            # Time point of last switchting event, we keep track of this since this is one continuous run of a solver
            t0 = 0
            life_time_list = []
            n_switching = 0

            # pointer to current spin directions
            spin_directions = system.get_spin_directions(p_state)

            simulation.start(
                p_state,
                simulation.METHOD_LLG,
                simulation.SOLVER_RK4,
                single_shot=True,
                n_iterations=2000000 * 60,
                n_iterations_log=10000000,
            )

            while simulation.running_on_image(p_state):
                spin_directions_prev = np.array(spin_directions)
                simulation.n_shot(p_state, N_SHOT)
                t = simulation.get_time(p_state)

                trajectory.append([t - t0, *spin_directions[0]])

                energy = system.get_energy(p_state)

                sz = spin_directions[0][2]
                if sz > 0.95:
                    n_switching += 1
                    life_time_list.append(t - t0)
                    print(
                        f"Found event {n_switching}/{N_EVENTS} with lifetime {t - t0} ps ( { int((t - t0) / DT) } steps )"
                    )

                    np.savetxt(
                        trajectory_folder / f"trajectory_{n_switching}",
                        trajectory,
                        header="t, sz, sy, sz",
                    )

                    t0 = t
                    trajectory = []
                    io.image_read(p_state, INITIAL)

                if n_switching == N_EVENTS:
                    simulation.stop_all(p_state)

            print("Found {} switching events".format(n_switching))
            avg_lifetime = np.mean(life_time_list)
            std_dev = np.std(life_time_list) / np.sqrt(n_switching)
            print("Lifetime = {} +- {}".format(avg_lifetime, std_dev))

            data.append([damping, temperature, avg_lifetime, std_dev, n_switching])

np.savetxt(DATA, data, header="damping temperature lifetime std_dev n_switching")
