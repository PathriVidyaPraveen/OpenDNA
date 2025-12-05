"""
Visualization module for OpenDNA.

Provides:
    - TunnelPlotter: heatmaps, classical-vs-quantum comparisons,
                     3D wavepacket evolution.

These plots are essential for analyzing tunneling dynamics and
checking whether quantum approximations track the exact solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  


class TunnelPlotter:


    def __init__(self, x_grid):

        self.x = x_grid
        self.N = len(x_grid)

    # Heatmap - position vs time vs probability

    def plot_heatmap(self, prob_dict, title="Tunneling Heatmap"):


        times = sorted(prob_dict.keys())
        prob_matrix = np.array([prob_dict[t] for t in times])

        plt.figure(figsize=(8, 6))
        plt.imshow(
            prob_matrix,
            aspect="auto",
            origin="lower",
            extent=[self.x[0], self.x[-1], times[0], times[-1]],
            cmap="viridis",
        )
        plt.colorbar(label="Probability")
        plt.xlabel("Position")
        plt.ylabel("Time")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # comparison - classical line vs quantum dots

    def plot_comparison(
        self,
        classical_probs,
        quantum_probs,
        time_point,
        title="Classical vs Quantum Tunneling",
    ):


        if time_point not in classical_probs:
            raise ValueError("time_point missing in classical_probs")
        if time_point not in quantum_probs:
            raise ValueError("time_point missing in quantum_probs")

        plt.figure(figsize=(8, 5))
        plt.plot(self.x, classical_probs[time_point], label="Classical", linewidth=2)
        plt.scatter(
            self.x,
            quantum_probs[time_point],
            color="red",
            s=15,
            label="Quantum",
            zorder=3,
        )
        plt.xlabel("Position")
        plt.ylabel("|ψ|²")
        plt.title(title + f" @ t = {time_point}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 3d evolution wavepacket over time

    def plot_3d_evolution(self, prob_dict, title="3D Wavepacket Evolution"):


        times = np.array(sorted(prob_dict.keys()))
        prob_matrix = np.array([prob_dict[t] for t in times])

        X, Y = np.meshgrid(self.x, times)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X,
            Y,
            prob_matrix,
            cmap=cm.plasma,
            linewidth=0,
            antialiased=True,
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Time")
        ax.set_zlabel("Probability")
        ax.set_title(title)

        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.show()
