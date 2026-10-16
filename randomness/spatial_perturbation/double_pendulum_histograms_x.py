"""
Generate x-coordinate histograms of double pendulum end-effector positions.

Uses 1000 pendulums to show statistical distribution of chaos over time.
"""

import numpy as np
import matplotlib.pyplot as plt

from double_pendulum_sampler import run_sampling, sample_positions_at_time


def main():
    # Parameters
    n_samples = 1000  # 1000 pendulums for better statistics
    L1 = 1.0
    L2 = 1.0
    dt = 0.01
    t_max = 20.0
    sample_times = np.arange(0, 21, 1)  # 0, 1, 2, ..., 20 seconds

    print(f"Simulating {n_samples} pendulums for {t_max}s...")
    print("This may take a moment...")
    x_initial, trajectories, _ = run_sampling(
        n_samples=n_samples,
        L1=L1,
        L2=L2,
        dt=dt,
        t_max=t_max,
        sample_time=t_max,
    )
    print("Simulation complete.")

    # Sample positions at each time point
    sampled_by_time = {}
    for t in sample_times:
        positions = sample_positions_at_time(trajectories, dt, t)
        x2 = positions[:, 2]
        sampled_by_time[t] = x2
        print(f"t={t:2d}s: x₂ ∈ [{x2.min():.3f}, {x2.max():.3f}], σ={x2.std():.3f}")

    # Create figure with histograms in a grid layout
    n_times = len(sample_times)
    n_cols = 3
    n_rows = (n_times + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    fig.suptitle(
        f"Double Pendulum End-Effector x-Distribution Over Time\n"
        f"({n_samples} pendulums, initial x ∈ [-2L, -L] at y=0)",
        fontsize=13,
    )

    reach = L1 + L2
    x_bins = np.linspace(-reach, reach, 30)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    for i, t in enumerate(sample_times):
        ax = axes_flat[i]
        x2 = sampled_by_time[t]

        ax.hist(x2, bins=x_bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(0, n_samples * 0.35)
        ax.set_title(f"t = {t}s", fontsize=11)
        ax.axvline(x2.mean(), color="red", lw=1.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls=":")

        # Add statistics
        ax.text(
            0.97, 0.95,
            f"μ={x2.mean():.2f}\nσ={x2.std():.2f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="none"),
        )

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("x position")

    # Hide unused subplots
    for i in range(n_times, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig("double_pendulum_histograms_x.png", dpi=150, transparent=True)
    print("\nSaved figure to double_pendulum_histograms_x.png")
    plt.show()


if __name__ == "__main__":
    main()
