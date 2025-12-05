"""
Noise utilities for OpenDNA.

Provides:
- add_biological_noise(circuit, rate):
      Inserts biologically-inspired stochastic phase noise or depolarizing noise
      between Trotter steps or key circuit segments.

- biological_noise_model(rate):
      Returns a Qiskit NoiseModel tuned to the gate set relevant for OpenDNA.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import RZGate, UnitaryGate


try:
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        phase_damping_error,
    )
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


# helper fucntions to detect trotter step boundaries (heuristic)

def _is_trotter_boundary(inst: Instruction) -> bool:

    if inst.name == "barrier":
        return True
    if isinstance(inst, UnitaryGate):
        return True
    
    if "paulievolution" in inst.name.lower():
        return True
    return False


# biological noise injection - inline circuit modification

def add_biological_noise(circuit: QuantumCircuit, rate: float = 0.05) -> QuantumCircuit:

    
    new_qc = QuantumCircuit(circuit.num_qubits)
    
    for reg in circuit.qregs:
        if reg not in new_qc.qregs:
            new_qc.add_register(reg)
    for reg in circuit.cregs:
        if reg not in new_qc.cregs:
            new_qc.add_register(reg)

    for instruction in circuit.data:
        inst = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        
        new_qc.append(inst, qargs, cargs)

        # Insert noise after boundaries
        if _is_trotter_boundary(inst):
            for qubit in qargs:
                
                if np.random.rand() < 0.7:
                    
                    theta = np.random.normal(0, rate)
                    new_qc.append(RZGate(theta), [qubit])
                else:
                    
                    phi = np.random.uniform(0, 2 * np.pi)
                    
                    rand_Rz = RZGate(phi * rate) 
                    new_qc.append(rand_Rz, [qubit])

    return new_qc


# biological noise model - backend

def biological_noise_model(rate: float = 0.02):

    if not AER_AVAILABLE:
        print("Warning: qiskit-aer not installed. Returning None for noise model.")
        return None

    nm = NoiseModel()

    dep = depolarizing_error(rate, 1)
    damp = phase_damping_error(rate)

    one_qubit_noise = dep.compose(damp)

    
    dep2 = depolarizing_error(rate * 1.5, 2)
    
    damp2 = damp.tensor(damp) 
    
    two_qubit_noise = dep2.compose(damp2)


    for gate in ["rz", "rx", "ry", "h", "u", "p"]:
        nm.add_all_qubit_quantum_error(one_qubit_noise, gate)

    nm.add_all_qubit_quantum_error(one_qubit_noise, "unitary")

    nm.add_all_qubit_quantum_error(two_qubit_noise, "cx")
    nm.add_all_qubit_quantum_error(two_qubit_noise, "ecr") 

    return nm