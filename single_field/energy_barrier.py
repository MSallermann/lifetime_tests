import numpy as np
import matplotlib.pyplot as plt
from spirit import constants
import numpy as np
from spirit_extras.plotting import Paper_Plot


def delta_e(field):
    K = 1
    delta_e_expected = (
        K - constants.mu_B * field + 0.25 * (constants.mu_B * field) ** 2 / K
    )
    return delta_e_expected


def max_T(field, factor=5):
    return delta_e(field) / (factor * constants.k_B)


def max_field(T, factor=5):
    K = 1
    a = 2 * K / constants.mu_B
    b = np.sqrt(4 * K * constants.k_B * T * factor) / constants.mu_B

    return a - b


if __name__ == "__main__":
    B = np.linspace(0, 5, 50)
    T = np.linspace(0.5, 5, 49)
    Z = np.zeros(shape=(len(B), len(T)))
    Z = [[delta_e(f) / (constants.k_B * t) for f in B] for t in T]

    # Paper_Plot()
    print(max_T(0))

    print(max_field(0.5))

    # print(delta_e(max_field(3)))
    # print(3*constants.k_B*5)

    plt.contourf(B, T, Z)
    plt.plot(B, max_T(B))
    plt.savefig("energy_barrier_b_vs_t.png", dpi=300)
    plt.show()
