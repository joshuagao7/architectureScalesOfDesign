"""
Sample x-coordinates at fixed time (t=10s) across 1000 different initial positions.

Shows how a uniform grid of initial conditions (x ∈ [-2L, -L]) maps to 
a chaotic distribution after 10 seconds of evolution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from double_pendulum_sim import generate_trajectories


def main():
    # Parameters
    n_samples = 20000
    L1 = 1.0
    L2 = 1.0
    m1 = 1.0
    m2 = 1.0
    g = 9.81
    dt = 0.01
    t_sample = 10.0  # Fixed sample time

    # Initial positions: uniform grid from -1.8L to -L
    x_initial = np.linspace(-1.8 * L1, -L1, n_samples)
    target_y = 0.0

    print(f"Simulating {n_samples} pendulums...")
    print(f"Initial x range: [{x_initial[0]:.3f}, {x_initial[-1]:.3f}]")
    print(f"Sampling at t = {t_sample}s")

    trajectories = generate_trajectories(
        x_initial, target_y, L1, L2, m1, m2, g, dt, t_sample, elbow="up"
    )

    # Extract final x positions at t=10s
    frame = int(t_sample / dt)
    x_final = np.array([traj[2][frame] for traj in trajectories])  # x2 at final frame

    print(f"\nResults at t = {t_sample}s:")
    print(f"  x_final range: [{x_final.min():.3f}, {x_final.max():.3f}]")
    print(f"  mean: {x_final.mean():.3f}")
    print(f"  std:  {x_final.std():.3f}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(
        f"Double Pendulum: 1000 Initial Positions → t = {t_sample}s\n"
        f"Initial x uniformly spaced in [-1.8L, -L] at y = 0",
        fontsize=13,
    )

    reach = L1 + L2

    # Plot 1: Initial vs Final x (scatter/line plot showing the mapping)
    ax1 = axes[0]
    ax1.plot(x_initial, x_final, ".", markersize=2, color="steelblue", alpha=0.6)
    ax1.set_xlabel("Initial x₀", fontsize=11)
    ax1.set_ylabel(f"Final x at t={t_sample}s", fontsize=11)
    ax1.set_title("Mapping: Initial Position → Final Position", fontsize=12)
    ax1.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlim(-1.9, -0.9)
    ax1.set_ylim(-reach, reach)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of initial positions (should be uniform)
    ax2 = axes[1]
    ax2.hist(x_initial, bins=50, color="gray", edgecolor="white", alpha=0.7)
    ax2.set_xlabel("x position", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Initial x Distribution (Uniform Grid)", fontsize=12)
    ax2.set_xlim(-reach, reach)
    ax2.axvline(x_initial.mean(), color="red", lw=1.5, ls="--", label=f"μ={x_initial.mean():.2f}")
    ax2.text(
        0.97, 0.95,
        f"σ={x_initial.std():.3f}",
        transform=ax2.transAxes,
        ha="right", va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none"),
    )

    # Plot 3: Histogram of final positions (should show chaos)
    ax3 = axes[2]
    ax3.hist(x_final, bins=50, color="coral", edgecolor="white", alpha=0.8)
    ax3.set_xlabel("x position", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title(f"Final x Distribution at t = {t_sample}s (Chaotic Spread)", fontsize=12)
    ax3.set_xlim(-reach, reach)
    ax3.axvline(x_final.mean(), color="red", lw=1.5, ls="--", label=f"μ={x_final.mean():.2f}")
    ax3.axvline(0, color="gray", lw=0.5, ls=":")
    ax3.text(
        0.97, 0.95,
        f"μ={x_final.mean():.3f}\nσ={x_final.std():.3f}",
        transform=ax3.transAxes,
        ha="right", va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none"),
    )

    plt.tight_layout()
    plt.savefig("sample_initial_positions.png", dpi=150, transparent=True)
    print("\nSaved figure to sample_initial_positions.png")

    # Also save the data to CSV
    data = np.column_stack([x_initial, x_final])
    np.savetxt(
        "sample_initial_positions.csv",
        data,
        delimiter=",",
        header="x_initial,x_final_10s",
        comments="",
    )
    print("Saved data to sample_initial_positions.csv")

    plt.show()


if __name__ == "__main__":
    main()
