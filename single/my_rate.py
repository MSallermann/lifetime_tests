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


def htst_rate_zero_mode_analytical(field, damping, T, diffusion=True):
    K = 1
    MU = 1
    from spirit import constants

    sx = constants.mu_B * field / (2 * K)
    sz = -np.sqrt(1 - sx**2)
    kbT = constants.k_B * T
    mu_field = constants.mu_B * field

    q = np.array([0, 0, 1])
    s = np.array([1, 0, 0])

    D = damping * constants.k_B * T * constants.mu_B * MU / constants.gamma
    gamma_prime = constants.gamma / ((1 + damping**2) * constants.mu_B * MU)

    delta_e_expected = K - constants.mu_B * field + 0.25 * (constants.mu_B * field) ** 2

    entropic_factor_expected = (
        2
        * np.pi
        * np.sqrt(2 * np.pi * constants.k_B * T) ** (-2)
        * (2 * K * -sz)
        * np.exp(-delta_e_expected / (constants.k_B * T))
    )

    variance_precession = 0
    variance_diffusion_prec = 0
    variance_diffusion_damping = gamma_prime**2 * (2 * D * (1 + damping**2))

    if diffusion:
        variance_expected = (
            variance_precession + variance_diffusion_damping + variance_diffusion_prec
        )
    else:
        variance_expected = variance_precession

    dynamic_factor_expected = np.sqrt(variance_expected / (2 * np.pi))

    return entropic_factor_expected * dynamic_factor_expected


def tst_rate(field, damping, T):
    K = 1
    MU = 1
    from spirit import constants
    from scipy.integrate import quad, nquad
    from scipy.stats import norm

    sx = constants.mu_B * field / (2 * K)
    sz = -np.sqrt(1 - sx**2)
    kbT = constants.k_B * T
    mu_field = constants.mu_B * field

    D = damping * constants.k_B * T * constants.mu_B * MU / constants.gamma
    gamma_prime = constants.gamma / ((1 + damping**2) * constants.mu_B * MU)

    ez = np.array([0, 0, 1])

    def s(phi):
        return np.array([np.cos(phi), np.sin(phi), 0])

    def b(xi1, xi2, xi3, phi):
        return -mu_field * np.cos(phi) * np.array([1, 0, 0]) + np.sqrt(
            2 * D
        ) * np.array([xi1, xi2, xi3])

    def E(phi):
        return -mu_field * np.cos(phi)

    def v_perp(xi1, xi2, xi3, phi):
        return -gamma_prime * ez.T @ np.cross(
            s(phi), b(xi1, xi2, xi3, phi)
        ) - gamma_prime * damping * ez.T @ np.cross(
            s(phi), np.cross(s(phi), b(xi1, xi2, xi3))
        )

    def Z_TS_density(phi):
        return np.exp(-1 / kbT * E(phi))

    Z_TS = quad(Z_TS_density, 0, 2 * np.pi)[0]
    print(f"{Z_TS = }")

    def theta(x):
        return 1 if x > 0 else 0

    def dyn_contribution_density(xi1, xi2, xi3, phi):
        return (
            1
            / Z_TS
            * (
                v_perp(xi1, xi2, xi3, phi)
                * theta(v_perp(xi1, xi2, xi3, phi))
                * norm.pdf(xi1)
                * norm.pdf(xi2)
                * norm.pdf(xi3)
                * np.exp(-1 / kbT * E(phi))
            )
        )

    