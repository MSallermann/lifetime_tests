import sys
import numpy as np

sys.path.insert(0, "../spirit/core/python")

input_cfg = "input_cube.cfg"

from spirit import (
    chain,
    state,
    htst,
    configuration,
    io,
    constants,
    simulation,
    system,
    tst_bennet,
)
from spirit.parameters import llg

with state.State(input_cfg, quiet=False) as p_state:

    configuration.plus_z(p_state)
    io.chain_read(p_state, "cube_chain.ovf")
    io.chain_write(p_state, "chain_bennet_initial.ovf")
    tst_bennet.calculate(p_state, 0, 1, 10)
