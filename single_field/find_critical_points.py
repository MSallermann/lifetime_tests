import numpy as np
from pathlib import Path

from spirit import state, io, simulation, hamiltonian, configuration, constants, system
from spirit_extras import calculation_folder, rate

THIS = Path(__file__).parent
INPUT = (THIS / "input.cfg").as_posix()

K = 1  # T
B = 5  # T mu is 1meV
folder = calculation_folder.Calculation_Folder(
    THIS / f"B_{B:.1f}_K_{K:.1f}", create=True, descriptor_file="descriptor.toml"
)

with state.State(INPUT) as p_state:
    folder["K"] = K
    folder["B"] = B
    spin_directions = system.get_spin_directions(p_state)
    hamiltonian.set_field(p_state, magnitude=B, direction=[1, 0, 0])
    hamiltonian.set_anisotropy(p_state, magnitude=K, direction=[0, 0, 1])

    configuration.minus_z(p_state)
    configuration.add_noise(p_state, 1e-3)
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_LBFGS_OSO)

    io.image_write(p_state, (folder / "minimum.ovf").as_posix())

    configuration.domain(p_state, [1, 0, 0])
    system.update_data(p_state)

    io.image_write(p_state, (folder / "sp.ovf").as_posix())

    res = rate.get_htst_quantities(
        p_state,
        file_min=folder / "minimum.ovf",
        file_sp=folder / "sp.ovf",
        workdir=Path(folder),
    )

    folder.copy_file(INPUT, "input.cfg")
    folder.to_desc()
