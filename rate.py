from spirit import constants
from numpy.typing import NDArray
import numpy as np

def skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Skew symmetric matrix from a vectors"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def delta(i: int, j: int) -> int:
    """Kronecker delta"""
    return 1 if i == j else 0


def compute_a_vector(q : NDArray[np.float64], s : NDArray[np.float64], H : NDArray[np.float64], alpha : float, mu : float):
    """Computes the 'a' vector"""
    gamma_prime = constants.gamma / ((1 + alpha**2) * constants.mu_B * mu)

    nos = int(H.shape[0] / 3)
    a = np.zeros(3 * nos)
    for i in range(nos):
        ai = np.zeros(3)
        for j in range(nos):q
            qj = q[3 * j : 3 * j + 3]
            sj = s[3 * j : 3 * j + 3]
            Hij = H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            ai += Hij @ np.cross(sj, qj)
        a[3 * i, 3 * i + 3] = ai
    return gamma_prime * a


def compute_b_vector(q, s, alpha, mu, T):
    gamma_prime = constants.gamma / ((1 + alpha**2) * constants.mu_B * mu)
    D = alpha * constants.k_B * T * constants.mu_B * mu / constants.gamma
    nos = int(s.shape[0] / 3)

    b = np.zeros(3 * nos)

    for i in range(nos):
        qi = q[3 * i : 3 * i + 3]
        si = s[3 * i : 3 * i + 3]
        b[3 * i : 3 * i + 3] = np.cross(si, qi) + alpha * qi

    return -gamma_prime * np.sqrt(2 * D) * b


def compute_Q_matrix(q, s, alpha, mu, T):
    gamma_prime = constants.gamma / ((1 + alpha**2) * constants.mu_B * mu)
    D = alpha * constants.k_B * T * constants.mu_B * mu / constants.gamma
    nos = int(s.shape[0] / 3)

    Q = np.zeros(shape=(3 * nos, 3 * nos))

    for i in range(nos):
        qi = q[3 * i : 3 * i + 3]
        Q[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = skew(qi)

    return -gamma_prime * np.sqrt(2 * D) * Q


def compute_dyn_contribution(q, s, H_2N, basis, alpha, mu, T):
    a = compute_a_vector(q, s, basis.T @ H_2N @ basis, alpha, mu)
    b = compute_b_vector(q, s, alpha, mu, T)
    Q = compute_Q_matrix(q, s, alpha, mu)

    # covariance matrix of boltzmann distribution projected to 3N space
    Sigma = constants.k_B * T * basis.T @ np.linalg.inv(H_2N) @ basis

    variance = a.T @ Sigma @ a + b.T @ b - np.trace(Sigma @ Q @ Q)

    dyn_factor = np.sqrt(variance / (2.0 * np.pi))
    return dyn_factor


def compute_entropic_factor(Vsp, Vm, N0m, N0sp, detHm, detHsp, delta_e, T):
    return (
        Vsp
        / Vm
        * np.sqrt(2 * np.pi * constants.k_B * T) ** (N0m - N0sp - 1)
        * np.sqrt(detHm / detHsp)
        * np.exp(-delta_e / (constants.k_B * T))
    )
