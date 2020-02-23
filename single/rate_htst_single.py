import sys
import numpy as np
sys.path.insert(0, "../spirit/core/python")

input_cfg = "input.cfg"

from spirit import chain, state, htst, configuration, io, constants, simulation, system
from spirit.parameters import llg

with state.State(input_cfg, quiet = False) as p_state:

    configuration.plus_z(p_state)
    io.image_read(p_state, "single_anis_min.ovf")

    chain.image_to_clipboard(p_state)
    chain.set_length(p_state, 2)

    io.image_read(p_state, "single_anis_SP.ovf", idx_image_inchain=1)
    io.chain_write(p_state, "chain_htst_initial.ovf")
    
    htst.calculate(p_state, 0, 1)
    temperature_exponent, me, Omega_0, s, volume_min, volume_sp, prefactor_dynamical, prefactor = htst.get_info(p_state)

    chain.update_data(p_state)

    dE = system.get_energy(p_state, idx_image=1)  - system.get_energy(p_state, idx_image=0)

    rate = prefactor * np.exp( -dE / (constants.k_B * llg.get_temperature(p_state)[0]) )

    print("Energy Barrier = {}".format(dE))
    print("Temperature    = {}".format(llg.get_temperature(p_state)[0]))
    print("Full transition rate = {} 1/ps".format( rate * 1e-12 ))
    print("life time = {} ps".format( 1/rate * 1e12 ))