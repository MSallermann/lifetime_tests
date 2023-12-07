import numpy as np
from pathlib import Path

# The Bohr Magneton [meV/T]
mu_B = 0.057883817555
# The vacuum permeability [T^2 m^3 / meV]
mu_0 = 2.0133545 * 1e-28
# The Boltzmann constant [meV/K]
k_B = 0.08617330350
# Planck constant [meV*ps/rad]
hbar = 0.6582119514
# Gyromagnetic ratio of electron [rad/(ps*T)]
# Also gives the Larmor precession frequency for electron
gamma = 0.1760859644


def rate(K, T, alpha, mu=1):
    delta_e = K
    kbT = k_B * T

    # return 2 * K / kbT * np.sqrt(alpha * kbT / np.pi) * np.exp(-delta_e / kbT)

    gamma_prime = gamma / ((1 + alpha**2) * mu_B * mu)

    D = alpha * kbT * mu_B * mu / gamma

    dynamical_contribution = np.sqrt(
        2 * D * gamma_prime**2 * (1 + alpha**2) / (2 * np.pi)
    )
    entropy_contribution = (
        2
        * np.pi
        * np.sqrt(2 * np.pi * k_B * T) ** (-2)
        * 2
        * K
        * np.exp(-delta_e / kbT)
    )

    return dynamical_contribution * entropy_contribution
