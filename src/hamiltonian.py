"""
Hamiltonian construction utilities for OpenDNA.

Provides:
- PotentialGenerator: constructors for common 1D potentials (harmonic, double-well, asymmetric).
- HamiltonianBuilder: builds kinetic energy via finite difference (sparse) and combines with potential.
- to_sparse_pauli_op: converts an NxN Hamiltonian (N == 2**n) into Qiskit's SparsePauliOp.

Notes:
- Uses scipy.sparse for efficient matrix handling.
- The SparsePauliOp conversion requires qiskit; if qiskit is missing a helpful error is raised.
- Converting to Pauli basis requires the matrix dimension to be a power of two.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags

__all__ = ["PotentialGenerator", "HamiltonianBuilder"]


class PotentialGenerator:


    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.x_min = float(cfg.get("x_min", -5.0))
        self.x_max = float(cfg.get("x_max", 5.0))
        self.n_points = int(cfg.get("n_points", 256))
        if self.n_points < 2:
            raise ValueError("n_points must be >= 2")
        self.x = np.linspace(self.x_min, self.x_max, self.n_points)
        self.dx = self.x[1] - self.x[0]

    def harmonic_potential(self, k: float = 1.0, x0: float = 0.0) -> np.ndarray:

        return 0.5 * k * (self.x - x0) ** 2

    def double_well_potential(self, a: float = 1.0, b: float = 1.0) -> np.ndarray:

        return a * self.x ** 4 - b * self.x ** 2

    def asymmetric_double_well(
        self, a: float = 1.0, b: float = 1.0, asymmetry: float = 0.1
    ) -> np.ndarray:

        return a * self.x ** 4 - b * self.x ** 2 + asymmetry * self.x

    def custom(self, func) -> np.ndarray:
        """
        Evaluate an arbitrary callable func(x) on the internal grid.
        """
        return np.asarray(func(self.x))


class HamiltonianBuilder:


    def __init__(self, potential: np.ndarray, dx: float, config: Optional[Dict] = None):
        if potential.ndim != 1:
            raise ValueError("potential must be a 1D numpy array")
        self.V = np.asarray(potential, dtype=float)
        self.N = self.V.size
        self.dx = float(dx)
        cfg = config or {}
        self.mass = float(cfg.get("mass", 1.0))
        self.hbar = float(cfg.get("hbar", 1.0))
        self.boundary = cfg.get("boundary", "dirichlet")
        if self.boundary not in ("dirichlet", "periodic"):
            raise ValueError("boundary must be 'dirichlet' or 'periodic'")

    def kinetic_operator(self) -> sp.spmatrix:

        N = self.N
        dx2 = self.dx ** 2
        coeff = -(self.hbar ** 2) / (2.0 * self.mass * dx2)

        main = np.full(N, -2.0)
        off = np.full(N - 1, 1.0)

        diags_data = [main, off, off]
        offsets = [0, -1, 1]
        lap = diags(diags_data, offsets, shape=(N, N), format="csr")

        if self.boundary == "periodic":
            
            lap = lap.tolil()
            lap[0, -1] = 1.0
            lap[-1, 0] = 1.0
            lap = lap.tocsr()

        T = coeff * lap
        return T

    def potential_operator(self) -> sp.spmatrix:
        """
        Construct potential energy operator as a diagonal matrix with entries V(x_i).
        """
        return diags(self.V, 0, format="csr")

    def hamiltonian(self) -> sp.spmatrix:
        """
        Return the full Hamiltonian H = T + V as a scipy.sparse CSR matrix.
        """
        T = self.kinetic_operator()
        Vop = self.potential_operator()
        H = T + Vop
        return H.tocsr()

    @staticmethod
    def is_power_of_two(n: int) -> bool:
        return (n & (n - 1) == 0) and n > 0

    def to_sparse_pauli_op(self, H_sparse: Optional[sp.spmatrix] = None):

        
        try:
            from qiskit.quantum_info import SparsePauliOp, Operator
        except Exception as e:
            raise ImportError(
                "Qiskit is required for to_sparse_pauli_op(). Install qiskit and retry. "
                f"Original error: {e}"
            )

        Hs = H_sparse if H_sparse is not None else self.hamiltonian()
        if not sp.issparse(Hs):
            # allow numpy arrays also
            H_dense = np.asarray(Hs, dtype=complex)
        else:
            # Convert to dense 
            H_dense = Hs.toarray().astype(complex)

        N = H_dense.shape[0]
        if H_dense.shape[0] != H_dense.shape[1]:
            raise ValueError("Hamiltonian must be a square matrix.")
        if not self.is_power_of_two(N):
            raise ValueError(
                f"Matrix dimension N={N} is not a power of two. "
                "Cannot directly map to qubit Pauli operators. "
                "Consider padding or using a different encoding."
            )

        
        op = Operator(H_dense)
        pauli_op = SparsePauliOp.from_operator(op)
        return pauli_op


