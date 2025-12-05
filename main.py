"""
OpenDNA Main Driver

Modes:
    --mode comparison   : Classical vs Quantum (clean)
    --mode noisy        : Quantum with biological noise

Usage:
    python main.py --mode comparison
    python main.py --mode noisy
"""

import argparse
import yaml
import numpy as np
import os

from qiskit_aer import AerSimulator

# Step modules
from src.hamiltonian import PotentialGenerator, HamiltonianBuilder
from src.classical_solver import ClassicalSchrodingerSolver
from src.quantum_solver import TrotterSolver
from src.noise_models import biological_noise_model
from src.visualizer import TunnelPlotter


#helper fucntions

def load_config(path="config/simulation_config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}. Please create it.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_potential(config):
    gen = PotentialGenerator({
        "x_min": config["x_min"],
        "x_max": config["x_max"],
        "n_points": config["grid_points"]
    })

    pot_type = config["potential_type"]

    if pot_type == "harmonic":
        return gen.harmonic_potential()
    elif pot_type == "double_well":
        p = config["potential_params"]
        return gen.double_well_potential(a=p["a"], b=p["b"])
    elif pot_type == "asymmetric_double_well":
        p = config["potential_params"]
        return gen.asymmetric_double_well(a=p["a"], b=p["b"], asymmetry=p["asymmetry"])
    else:
        raise ValueError(f"Unknown potential type: {pot_type}")


def classical_simulation(H_dense, psi0, times):
    classical = ClassicalSchrodingerSolver(H_dense)
    psi_exact = classical.evolve(psi0, times)
    probs = classical.probabilities(psi_exact)
    return probs


def quantum_simulation(
    sparse_pauli_op, psi0, times, backend, apply_noise=False, noise_rate=0.05
):
    
    solver = TrotterSolver(sparse_pauli_op, backend, trotter_order=2)

    results = solver.evolve(psi0, times)

    
    probs = {}
    dim = 2**sparse_pauli_op.num_qubits

    for t, sv_or_counts in results.items():
        if hasattr(sv_or_counts, "probabilities"):  # Statevector object
            probs[t] = sv_or_counts.probabilities()
        elif isinstance(sv_or_counts, np.ndarray):  # Raw numpy array
             probs[t] = np.abs(sv_or_counts) ** 2
        else:  
            vec = np.zeros(dim)
            total_shots = sum(sv_or_counts.values())
            for bitstr, count in sv_or_counts.items():
                
                idx = int(bitstr, 2)
                if idx < dim:
                    vec[idx] = count / total_shots
            probs[t] = vec

    return probs


# main execution

def main(mode):
    print(f"\nOpenDNA Simulation | Mode: {mode.upper()}\n" + "="*40)
    config = load_config()

    # Grid & time steps
    grid_points = config["grid_points"]
    num_qubits = config["num_qubits"]
    if grid_points != 2 ** num_qubits:
        raise ValueError(f"grid_points ({grid_points}) must equal 2^{num_qubits}")

    times = np.linspace(
        config["time_start"], config["time_end"], config["num_time_steps"]
    )

    # Build potential
    print(f"Generating {config['potential_type']} potential...")
    V = build_potential(config)
    x_grid = np.linspace(config["x_min"], config["x_max"], grid_points)

    # Initial state: localized Gaussian at left well (approx -1.2 Å)
    
    x0_start = -1.2
    psi0 = np.exp(-(x_grid - x0_start) ** 2 / (2 * 0.4**2))
    psi0 = psi0 / np.linalg.norm(psi0)

    # Hamiltonian
    print("Constructing Hamiltonian and Pauli Operators...")
    H_builder = HamiltonianBuilder(
        V, dx=x_grid[1] - x_grid[0],
        config={"mass": config["mass"], "hbar": config["hbar"]}
    )
    H_sparse = H_builder.hamiltonian()
    H_dense = H_sparse.toarray()

    # Convert to SparsePauliOp for quantum
    H_pauli = H_builder.to_sparse_pauli_op(H_sparse)

    # Visualizer
    plotter = TunnelPlotter(x_grid)

    # Backend Setup
    if mode == "noisy":
        print(f"Configuring Noisy Backend (Rate: {config['noise_rate']})...")
        noise = biological_noise_model(config["noise_rate"])
        if noise is None:
            print("     Warning: Noise model failed (missing qiskit-aer). Falling back to ideal.")
            backend = AerSimulator()
        else:
            backend = AerSimulator(noise_model=noise)
    else:
        print("Configuring Ideal Statevector Backend...")
        backend = AerSimulator(method="statevector")

    # comparison - classical vs quantum
    if mode == "comparison":
        print("Running CLASSICAL Exact Diagonalization...")
        classical_probs = classical_simulation(H_dense, psi0, times)

        print("Running QUANTUM (Trotter) Simulation...")
        quantum_probs = quantum_simulation(
            H_pauli, psi0, times, backend, apply_noise=False
        )

        print("Generating Plots...")
        plotter.plot_heatmap(classical_probs, title="Classical Tunneling Heatmap (Ground Truth)")
        
        # plotting quantum heatmap for visual checking
        plotter.plot_heatmap(quantum_probs, title="Quantum Tunneling Heatmap (Trotter)")

        mid_t = times[len(times) // 2]
        plotter.plot_comparison(
            classical_probs, quantum_probs, mid_t,
            title=f"Classical vs Quantum @ t={mid_t:.2f}"
        )

        plotter.plot_3d_evolution(quantum_probs, title="3D Quantum Wavepacket Evolution")

    # noisy - quantum only with biological noise model
    elif mode == "noisy":
        print("Running NOISY quantum evolution...")

        quantum_probs_noisy = quantum_simulation(
            H_pauli, psi0, times, backend,
            apply_noise=True, noise_rate=config["noise_rate"]
        )

        print("Plotting noisy tunneling heatmap...")
        plotter.plot_heatmap(
            quantum_probs_noisy, title=f"Noisy Quantum Tunneling (Rate={config['noise_rate']})"
        )
        
        plotter.plot_3d_evolution(
            quantum_probs_noisy, title="3D Noisy Evolution (Decoherence)"
        )

    else:
        raise ValueError("Unknown mode: " + mode)

    print("\nSimulation Complete.")


# argparse entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenDNA driver")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["comparison", "noisy"],
        default="comparison",
        help="Run classical vs quantum comparison OR noisy quantum simulation",
    )
    args = parser.parse_args()

    main(args.mode)