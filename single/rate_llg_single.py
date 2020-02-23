import sys
sys.path.insert(0, "../spirit/core/python")
import numpy as np 
input_cfg = "input.cfg"

from spirit import chain, state, htst, configuration, io, simulation, quantities, system
from spirit.parameters import llg

# Function that checks if system is still in metastable antiparallel trimer state, returns true if still in antiparallel trimer state
def check_single(directions):
    dz = directions[0][2]
    if(dz < 0):
        # decay into parallel
        return 0
    return 2


data = []

for damping in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    with state.State(input_cfg, quiet = True) as p_state:
        llg.set_damping(p_state, damping)
        io.image_read(p_state, "single_anis_min.ovf") # Load metastable antiparallel trimer configuration

        life_time_list = []
        n_switching = 0
        n_par  = 0
        n_apar = 0
        t0 = 0

        spin_directions = system.get_spin_directions(p_state)
        simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_DEPONDT, single_shot=True)

        while(simulation.running_on_image(p_state)):
            simulation.n_shot(p_state, 10)
            t  = simulation.get_time(p_state)
            spin_directions = system.get_spin_directions(p_state)

            check = check_single(spin_directions)
            if( check == 0 or check == 1 ):
                if(check==0):
                    n_par+=1
                if(check==1):
                    n_apar+=1
                n_switching += 1
                life_time_list.append(t-t0)
                t0=t
                io.image_read(p_state, "single_anis_min.ovf")

        print("Found {} switching events".format(n_switching))
        avg_lifetime = np.mean(life_time_list)
        std_dev      = np.std(life_time_list) / np.sqrt(n_switching)
        print("Lifetime = {} +- {}".format(avg_lifetime, std_dev))

        data.append([damping, avg_lifetime, std_dev, n_switching, n_par, n_apar])

np.savetxt("data.txt", data)

