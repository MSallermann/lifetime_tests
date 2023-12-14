from spirit_extras import rate
from pathlib import Path
import numpy as np
from spirit import state, hamiltonian, constants, system, io

THIS = Path(__file__).parent
INPUT = THIS / "input.cfg"

MU = 1
K = 1


def htst_rate(field, damping, T):
    WORKDIR = THIS / "temp"

    with state.State(INPUT.as_posix()) as p_state:
        hamiltonian.set_field(p_state, field, [1, 0, 0])

        sx = constants.mu_B * field / (2 * K)
        sz = -np.sqrt(1 - sx**2)
        spins = system.get_spin_directions(p_state)
        spins[0] = [sx, 0, sz]
        system.update_data(p_state)
        e_min = system.get_energy(p_state)

        io.image_write(p_state, (WORKDIR / "min.ovf").as_posix())
        spins[0] = [1, 0, 0]
        system.update_data(p_state)
        e_sp = system.get_energy(p_state)

        io.image_write(p_state, (WORKDIR / "sp.ovf").as_posix())
        res = rate.get_htst_quantities(
            p_state,
            file_min=WORKDIR / "min.ovf",
            file_sp=WORKDIR / "sp.ovf",
            workdir=WORKDIR,
        )

    kbT = constants.k_B * T
    mu_field = constants.mu_B * field

    q = np.array([0, 0, 1])
    s = np.array([1, 0, 0])

    H = res.hessian_sp_2n
    basis = res.basis_sp

    print(H.shape)
    print(basis.shape)

    D = damping * constants.k_B * T * constants.mu_B * MU / constants.gamma
    gamma_prime = constants.gamma / ((1 + damping**2) * constants.mu_B * MU)
    a_vector = rate.compute_a_vector(q, s, basis @ H @ basis.T, damping, MU, T)
    b_vector = rate.compute_b_vector(q, s, damping, MU, T)
    Q_matrix = rate.compute_Q_matrix(q, s, damping, MU, T)

    a_vector_expected = gamma_prime * constants.mu_B * field * np.array([0, 1, 0])
    print(f"{a_vector = }")
    print(f"{a_vector_expected = }")

    b_vector_expected = -gamma_prime * np.sqrt(2 * D) * np.array([0, 1, damping])
    print(f"{ b_vector = }")
    print(f"{ b_vector_expected = }")

    Q_matrix_expected = (
        -gamma_prime * np.sqrt(2 * D) * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    )
    print(f"{Q_matrix = }")
    print(f"{Q_matrix_expected = }")

    print("spin_min")
    print(f"{sx = }")
    print(f"{sz = }")

    print("energies")
    print(f"{e_sp = }")
    print(f"{e_min = }")
    print(f"{e_sp - e_min = }")

    e_sp_expected = -constants.mu_B * field
    print(f"{e_sp_expected = }")
    e_min_expected = -K - 0.25 * (constants.mu_B * field) ** 2 / K
    print(f"{e_min_expected = }")
    delta_e_expected = K - constants.mu_B * field + 0.25 * (constants.mu_B * field) ** 2
    print(f"{delta_e_expected = }")

    print(f"{res.det_sp = }")
    det_sp_expected = constants.mu_B * field
    print(f"{det_sp_expected = }")

    print(f"{res.det_min = }")
    det_min_expected = 4 * K**2 * sz**2
    print(f"{det_min_expected = }")

    Vsp = 1
    Vm = 1
    N0m = 0
    N0sp = 0
    det_min = res.det_min
    det_sp = res.det_sp

    dynamic_factor = rate.compute_dyn_contribution(
        q=q, s=s, H_2N=res.hessian_sp_2n, basis=res.basis_sp, alpha=damping, mu=MU, T=T
    )

    entropic_factor = rate.compute_entropic_factor(
        Vsp=Vsp,
        Vm=Vm,
        N0m=N0m,
        N0sp=N0sp,
        det_min=det_min,
        det_sp=det_sp,
        delta_e=e_sp - e_min,
        T=T,
    )

    entropic_factor_expected = (
        np.sqrt(2 * np.pi * constants.k_B * T) ** (-1)
        * (2 * K * -sz)
        / np.sqrt(constants.mu_B * field)
        * np.exp(-delta_e_expected / (constants.k_B * T))
    )

    variance_expected = gamma_prime**2 * (
        kbT * mu_field + kbT * 2 * D / mu_field + 2 * D * (1 + damping**2)
    )
    dynamic_factor_expected = np.sqrt(variance_expected / (2 * np.pi))
    print(f"dynamic_factor = {dynamic_factor}")
    print(f"dynamic_factor_expected = {dynamic_factor_expected}")

    print(f"entropic_factor = {entropic_factor}")
    print(f"entropic_factor_expected = {entropic_factor_expected}")

    htst_rate = entropic_factor * dynamic_factor
    print(htst_rate)
    return htst_rate


def htst_rate_analytical(field, damping, T, diffusion=True):
    sx = constants.mu_B * field / (2 * K)
    sz = -np.sqrt(1 - sx**2)
    kbT = constants.k_B * T
    mu_field = constants.mu_B * field

    q = np.array([0, 0, 1])
    s = np.array([1, 0, 0])

    D = damping * constants.k_B * T * constants.mu_B * MU / constants.gamma
    gamma_prime = constants.gamma / ((1 + damping**2) * constants.mu_B * MU)

    delta_e_expected = K - constants.mu_B * field + 0.25 * (constants.mu_B * field) ** 2 / K

    entropic_factor_expected = (
        np.sqrt(2 * np.pi * constants.k_B * T) ** (-1)
        * (2 * K * -sz)
        / np.sqrt(constants.mu_B * field)
        * np.exp(-delta_e_expected / (constants.k_B * T))
    )

    variance_precession = gamma_prime**2 * (kbT * mu_field)
    variance_diffusion_prec = gamma_prime**2 * (kbT * 2 * D / mu_field)
    variance_diffusion_damping = gamma_prime**2 * (2 * D * (1 + damping**2))

    if diffusion:
        variance_expected = (
            variance_precession + variance_diffusion_damping + variance_diffusion_prec
        )
    else:
        variance_expected = variance_precession

    dynamic_factor_expected = np.sqrt(variance_expected / (2 * np.pi))

    return entropic_factor_expected * dynamic_factor_expected


def tst_rate_analytical(field, damping, T, diffusion=True):
    from scipy.integrate import quad

    sx = constants.mu_B * field / (2 * K)
    sz = -np.sqrt(1 - sx**2)
    kbT = constants.k_B * T
    mu_field = constants.mu_B * field

    q = np.array([0, 0, 1])
    s = np.array([1, 0, 0])

    D = damping * constants.k_B * T * constants.mu_B * MU / constants.gamma
    gamma_prime = constants.gamma / ((1 + damping**2) * constants.mu_B * MU)

    delta_e_expected = K - constants.mu_B * field + 0.25 * (constants.mu_B * field) ** 2 / K

    entropic_factor_expected = (
        np.sqrt(2 * np.pi * constants.k_B * T) ** (-1)
        * (2 * K * -sz)
        / np.sqrt(constants.mu_B * field)
        * np.exp(-delta_e_expected / (constants.k_B * T))
    )

    variance_precession = gamma_prime**2 * (kbT * mu_field)
    variance_diffusion_prec = gamma_prime**2 * (kbT * 2 * D / mu_field)
    variance_diffusion_damping = gamma_prime**2 * (2 * D * (1 + damping**2))

    if diffusion:
        variance_expected = (
            variance_precession + variance_diffusion_damping + variance_diffusion_prec
        )
    else:
        variance_expected = variance_precession

    dynamic_factor_expected = np.sqrt(variance_expected / (2 * np.pi))

    return entropic_factor_expected * dynamic_factor_expected


def htst_rate_zero_mode_analytical(field, damping, T, diffusion=True):
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
        return mu_field * np.sin(phi) * np.array([1, 0, 0]) + np.sqrt(2 * D) * np.array(
            [xi1, xi2, xi3]
        )

    def E(phi):
        return -mu_field * np.cos(phi)

    def v_perp(xi1, xi2, xi3, phi):
        _s = s(phi)
        _b = b(xi1, xi2, xi3, phi)
        # return -gamma_prime *
        return -gamma_prime * ez.T @ np.cross(
            _s, _b
        ) - gamma_prime * damping * ez.T @ np.cross(_s, np.cross(_s, _b))

    def v_perp(xi1, xi2, xi3, phi):
        return (
            -gamma_prime * np.sin(phi) * np.cos(phi) * mu_field
            - gamma_prime * (xi1 * np.sin(phi) + xi2 * np.cos(phi))
            - gamma_prime * damping * xi3
        )
        # _s = s(phi)
        # _b = b(xi1, xi2, xi3, phi)
        # return -gamma_prime *
        # return -gamma_prime * ez.T @ np.cross(
        #     _s, _b
        # ) - gamma_prime * damping * ez.T @ np.cross(_s, np.cross(_s, _b))

    def Z_TS_density(phi):
        return np.exp(-1 / kbT * E(phi))

    Z_TS = quad(Z_TS_density, 0, 2 * np.pi)[0]
    print(f"{Z_TS = }")

    det_min_expected = 4 * K**2 * sz**2
    e_min_expected = -K - 0.25 * (constants.mu_B * field) ** 2 / K
    Z_MIN = (
        np.sqrt(2 * np.pi * kbT) ** (2)
        * 1.0
        / np.sqrt(det_min_expected)
        * np.exp(-e_min_expected / kbT)
    )
    print(f"{Z_MIN = }")
    entropy_contribution = Z_TS / Z_MIN
    print(f"{entropy_contribution = }")

    def theta(x):
        return 1 if x > 0 else 0

    def dyn_contribution_density(xi1, xi2, xi3, phi):
        v_p = v_perp(xi1, xi2, xi3, phi)
        if v_p < 0:
            return 0

        res = (
            1.0
            / Z_TS
            * (
                v_p
                * norm.pdf(xi1)
                * norm.pdf(xi2)
                * norm.pdf(xi3)
                * np.exp(-1 / kbT * E(phi))
            )
        )
        return res

    ranges = [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [0, 2 * np.pi]]
    ranges = [[-1, 1], [-1, 1], [-1, 1], [0, 2 * np.pi]]

    dyn_contribution = nquad(dyn_contribution_density, ranges)[0]

    rate = Z_TS / Z_MIN * dyn_contribution

    return rate

if __name__ == "__main__":

    field = 5
    damping = 0.3
    T = 2

    htst_rate(field, damping, T)

    tst_rate(field, damping, T)
