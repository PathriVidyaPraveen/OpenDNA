"""
Exact classical time evolution for OpenDNA.

This module provides:
    - ClassicalSchrodingerSolver: solves i dψ/dt = H ψ
      using matrix exponentials exp(-i H t).

Ground truth for benchmarking vs. quantum solvers.
"""

import numpy as np
from scipy.linalg import expm


class ClassicalSchrodingerSolver:


    def __init__(self, H):
        if hasattr(H, "toarray"):
            self.H = H.toarray().astype(complex)
        else:
            self.H = np.asarray(H, dtype=complex)

        if self.H.shape[0] != self.H.shape[1]:
            raise ValueError("Hamiltonian must be square")

        self.N = self.H.shape[0]

    def evolve(self, psi0, time_points):


        psi0 = np.asarray(psi0, dtype=complex)
        if psi0.size != self.N:
            raise ValueError("psi0 dimension mismatch")

        results = {}

        for t in time_points:
            U_t = expm(-1j * self.H * t)
            psi_t = U_t @ psi0
            results[t] = psi_t

        return results

    def probabilities(self, psi_dict):

        return {t: np.abs(psi) ** 2 for t, psi in psi_dict.items()}
