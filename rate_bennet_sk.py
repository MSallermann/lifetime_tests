import sys
import numpy as np
sys.path.insert(0, "../spirit/core/python")

input_cfg = "input_sk.cfg"

from spirit import chain, state, htst, configuration, io, constants, simulation, system, tst_bennet
from spirit.parameters import llg

with state.State(input_cfg, quiet = False) as p_state:

    configuration.plus_z(p_state)
    io.image_read(p_state, "skyrmion.ovf")

    chain.image_to_clipboard(p_state)
    chain.set_length(p_state, 2)

    io.image_read(p_state, "skyrmion_sp.ovf", idx_image_inchain=1)
    io.chain_write(p_state, "chain_bennet_initial.ovf")
    
    tst_bennet.calculate(p_state, 0, 1)