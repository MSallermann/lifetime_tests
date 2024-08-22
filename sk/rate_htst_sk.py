import sys
import numpy as np

sys.path.insert(0, "../spirit/core/python")

input_cfg = "input_sk.cfg"

from spirit import (
    chain,
    state,
    htst,
    configuration,
    io,
    constants,
    simulation,
    system,
    geometry,
)
from spirit.parameters import llg

with state.State(input_cfg, quiet=False) as p_state:

    configuration.plus_z(p_state)

    chain.image_to_clipboard(p_state)
    chain.set_length(p_state, 2)

    io.image_read(p_state, "skyrmion_sp.ovf", idx_image_inchain=1)
    io.chain_write(p_state, "chain_htst_initial.ovf")

    htst.calculate(p_state, 0, 1, n_eigenmodes_keep=10)
    (
        temperature_exponent,
        me,
        Omega_0,
        s,
        volume_min,
        volume_sp,
        prefactor_dynamical,
        prefactor,
    ) = htst.get_info(p_state)

    chain.update_data(p_state)

    dE = system.get_energy(p_state, idx_image=1) - system.get_energy(
        p_state, idx_image=0
    )

    rate = prefactor * np.exp(-dE / (constants.k_B * llg.get_temperature(p_state)[0]))

    nos = len(geometry.get_positions(p_state))

    eigen_vectors_min = np.array(htst.get_eigenvectors_min(p_state))

    # print( np.linalg.norm(eigen_vectors_min[10]) )

    print("Energy Barrier = {} meV".format(dE))
    print("Temperature    = {} K".format(llg.get_temperature(p_state)[0]))
    print("Full transition rate = {} 1/ps".format(rate * 1e-12))
    print("Life time            = {} ps".format(1 / rate * 1e12))
