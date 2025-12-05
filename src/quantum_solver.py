"""
Quantum solver layer for OpenDNA.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List

import numpy as np
from scipy.linalg import expm as scipy_expm

# Qiskit 1.0 Core Imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

# Qiskit Algorithms Imports
try:
    from qiskit_algorithms import VarQTE, TimeEvolutionProblem
    from qiskit_algorithms.optimizers import SPSA, COBYLA
    VARQTE_AVAILABLE = True
except ImportError:
    VARQTE_AVAILABLE = False
    SPSA = None
    COBYLA = None


# abstract base class

class QuantumSolver(ABC):
    def __init__(self, H: SparsePauliOp, backend):
        self.H = H
        self.backend = backend

    @abstractmethod
    def evolve(self, psi0: Union[np.ndarray, Statevector], time_points: List[float]):
        pass


# trotter solver

class TrotterSolver(QuantumSolver):
    def __init__(self, H: SparsePauliOp, backend, trotter_order: int = 1):
        super().__init__(H, backend)
        self.synthesis = SuzukiTrotter(order=trotter_order)

    def evolve(self, psi0, time_points):
        results = {}
        num_qubits = self.H.num_qubits

        if not isinstance(psi0, Statevector):
            psi0 = Statevector(psi0)

        safe_basis = ['u', 'cx', 'id']

        for t in time_points:
            if t == 0:
                results[0.0] = psi0
                continue

            # create evolution circuit
            evo_gate = PauliEvolutionGate(self.H, time=t, synthesis=self.synthesis)
            qc = QuantumCircuit(num_qubits)
            qc.prepare_state(psi0)
            qc.append(evo_gate, range(num_qubits))

            qc_safe = transpile(qc, basis_gates=safe_basis, optimization_level=1)

            is_statevector = False
            if hasattr(self.backend, 'name') and 'statevector' in self.backend.name:
                is_statevector = True
            elif hasattr(self.backend, 'options'):
                 if self.backend.options.get('method') == 'statevector':
                     is_statevector = True

            if is_statevector:
                qc_safe.save_statevector()
                job = self.backend.run(qc_safe)
                try:
                    sv = job.result().get_statevector(qc_safe)
                    results[t] = sv
                except Exception:
                    results[t] = job.result().data()['statevector']
            else:
                qc_safe.measure_all()

                job = self.backend.run(qc_safe, shots=2048)
                results[t] = job.result().get_counts()

        return results


#variational solver

class VariationalSolver(QuantumSolver):
    def __init__(
        self,
        H: SparsePauliOp,
        backend,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Any] = None,
        num_layers: int = 2,
    ):
        super().__init__(H, backend)
        self.num_qubits = H.num_qubits

        if ansatz is None:
            self.ansatz = EfficientSU2(self.num_qubits, reps=num_layers)
        else:
            self.ansatz = ansatz

        if SPSA is not None:
            self.optimizer = optimizer or SPSA(maxiter=50)
        else:
            self.optimizer = None

    def evolve(self, psi0, time_points):
        return self._evolve_custom_vqe(psi0, time_points)

    def _evolve_custom_vqe(self, psi0, time_points):
        results = {}
        if self.optimizer is None:
            raise ImportError("qiskit-algorithms not installed.")

        if not isinstance(psi0, Statevector):
            psi0 = Statevector(psi0)
            
        params = np.zeros(self.ansatz.num_parameters)
        safe_basis = ['u', 'cx', 'id']

        for t in time_points:
            H_mat = self.H.to_matrix()
            U_mat = scipy_expm(-1j * t * H_mat)
            target_sv = Statevector(U_mat @ psi0.data)

            def cost_function(p):
                bound_ansatz = self.ansatz.assign_parameters(p)
                current_sv = Statevector(bound_ansatz)
                fid = state_fidelity(target_sv, current_sv)
                return 1.0 - fid

            result = self.optimizer.minimize(cost_function, x0=params)
            params = result.x 

            final_ansatz = self.ansatz.assign_parameters(params)
            
            # One-pass transpile here too
            qc_safe = transpile(final_ansatz, basis_gates=safe_basis)

            if hasattr(self.backend, 'name') and 'statevector' in self.backend.name:
                qc_safe.save_statevector()
                job = self.backend.run(qc_safe)
                results[t] = job.result().get_statevector(qc_safe)
            else:
                qc_safe.measure_all()
                job = self.backend.run(qc_safe, shots=2048)
                results[t] = job.result().get_counts()

        return results