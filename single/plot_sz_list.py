import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

THIS = Path(__file__).parent

damping = 0.3
temperature = 3.0
trajectory_file = (
    THIS / "trajectories" / f"damping_{damping:.3f}_temperature_{temperature:.3f}"
)

data = np.loadtxt(trajectory_file)
print(len(data))
plt.plot(data[:, 0], data[:, -1])
plt.axhline(0)
plt.show()
