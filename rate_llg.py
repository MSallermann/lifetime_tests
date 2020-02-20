import sys
sys.path.insert(0, "../spirit/core/python")
import numpy as np 
input_cfg = "input.cfg"

from spirit import chain, state, htst, configuration, io, simulation, quantities
from spirit.parameters import llg

data = []
# for damping in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
for damping in [0.3]:
    with state.State(input_cfg, quiet = False) as p_state:
        llg.set_damping(p_state, damping)
        configuration.plus_z(p_state)
        mz_list = []
        t_list  = []
        life_time_list = []
        n_switching = 0
        simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_DEPONDT, single_shot=True)
        t0 = 0
        while(simulation.running_on_image(p_state)):
            simulation.single_shot(p_state)
            t  = simulation.get_time(p_state)
            mz = quantities.get_magnetization(p_state)[2]

            t_list.append(t)
            mz_list.append(mz)

            if(mz < 0):
                n_switching += 1
                life_time_list.append(t-t0)
                t0=t
                configuration.plus_z(p_state)

        print("Found {} switching events".format(n_switching))
        avg_lifetime = np.mean(life_time_list)
        std_dev      = np.std(life_time_list) / np.sqrt(n_switching)
        print("Lifetime = {} +- {}".format(avg_lifetime, std_dev))

        data.append([damping, avg_lifetime, std_dev, n_switching])
        # np.savetxt("t.txt", t_list)
        # np.savetxt("mz.txt", mz_list)
        # np.savetxt("life_time_list.txt", life_time_list)
np.savetxt("data.txt", data)