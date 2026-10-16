"""
Generate histograms of double pendulum end-effector positions at multiple time points.

Shows how initially similar configurations diverge into chaotic distributions over time.
Output: SVG written to final assets/ (vector, not PNG wrapper).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from double_pendulum_sampler import run_sampling, sample_positions_at_time

# Output to final assets (relative to repo root)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(_REPO_ROOT, "final assets")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "double_pendulum_histograms.svg")


def main():
    # Parameters
    n_samples = 100
    L1 = 1.0
    L2 = 1.0
    dt = 0.01
    t_max = 10.0
    sample_times = np.arange(0, 7, 1)  # 0, 1, 2, 3, 4, 5, 6 seconds

    print(f"Simulating {n_samples} pendulums...")
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
        sampled_by_time[t] = positions
        print(f"t={t:2d}s: x2 range [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}], "
              f"y2 range [{positions[:, 3].min():.3f}, {positions[:, 3].max():.3f}]")

    # Create a single-row skyline of histograms
    n_times = len(sample_times)
    n_cols = n_times * 2  # x and y histograms per time step
    fig_width = max(12, n_cols * 0.9)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 2.5))

    reach = L1 + L2
    x_bins = np.linspace(-reach, reach, 25)
    y_bins = np.linspace(-reach, reach, 25)

    for i, t in enumerate(sample_times):
        positions = sampled_by_time[t]
        x2 = positions[:, 2]  # End effector x
        y2 = positions[:, 3]  # End effector y

        # X histogram
        ax_x = axes[2 * i]
        counts_x, _, _ = ax_x.hist(x2, bins=x_bins, color="0.3", edgecolor="none")
        ax_x.set_xlim(-reach, reach)
        ax_x.set_ylim(0, counts_x.max() * 1.1)  # Scale to max count with small margin
        ax_x.margins(x=0)
        ax_x.axis("off")

        # Y histogram
        ax_y = axes[2 * i + 1]
        counts_y, _, _ = ax_y.hist(y2, bins=y_bins, color="0.3", edgecolor="none")
        ax_y.set_xlim(-reach, reach)
        ax_y.set_ylim(0, counts_y.max() * 1.1)  # Scale to max count with small margin
        ax_y.margins(x=0)
        ax_y.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(
        OUTPUT_PATH,
        format="svg",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()
    print(f"\nSaved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
