import sys

sys.path.append("../spirit/core/python")

from spirit import state, io, tst_bennet, htst, configuration, chain

with state.State("input.cfg") as p_state:

    configuration.plus_z(p_state)
    io.image_read(p_state, "single_anis_min.ovf")

    chain.image_to_clipboard(p_state)
    chain.set_length(p_state, 2)

    io.image_read(p_state, "single_anis_SP.ovf", idx_image_inchain=1)
    io.chain_write(p_state, "chain_htst_initial.ovf")

    tst_bennet.calculate(p_state, 0, 1)
