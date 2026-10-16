"""
Spatial autocorrelation of double pendulum final positions.

Shows how correlated the final x-positions are for pendulums with nearby
initial conditions. This reveals the Lyapunov divergence - nearby initial
conditions rapidly decorrelate.

x-axis: spatial lag (difference in initial x position)
y-axis: autocorrelation of final x positions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from double_pendulum_sim import generate_trajectories


def spatial_autocorrelation(x_final, max_lag=None):
    """
    Compute spatial autocorrelation for ordered sequence of final positions.
    
    For each lag k, computes correlation between x_final[i] and x_final[i+k].
    This measures how similar final positions are for pendulums that started
    k steps apart in initial condition space.
    """
    n = len(x_final)
    if max_lag is None:
        max_lag = n // 4
    
    x_centered = x_final - np.mean(x_final)
    var = np.sum(x_centered ** 2)
    
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.sum(x_centered[:-lag] * x_centered[lag:]) / var
    
    return acf


def main():
    # Parameters
    n_samples = 1000
    L1 = 1.0
    L2 = 1.0
    m1 = 1.0
    m2 = 1.0
    g = 9.81
    dt = 0.01
    t_sample = 10.0

    # Initial positions: uniform grid from -2L to -L
    x_initial = np.linspace(-2 * L1, -L1, n_samples)
    target_y = 0.0
    dx = x_initial[1] - x_initial[0]  # Spacing between initial conditions

    print(f"Simulating {n_samples} pendulums...")
    print(f"Initial x range: [{x_initial[0]:.3f}, {x_initial[-1]:.3f}]")
    print(f"Initial x spacing: Δx = {dx:.6f}")
    print(f"Sampling at t = {t_sample}s")

    trajectories = generate_trajectories(
        x_initial, target_y, L1, L2, m1, m2, g, dt, t_sample, elbow="up"
    )

    # Extract final x positions
    frame = int(t_sample / dt)
    x_final = np.array([traj[2][frame] for traj in trajectories])

    print(f"\nFinal x range: [{x_final.min():.3f}, {x_final.max():.3f}]")

    # Compute spatial autocorrelation
    max_lag = n_samples // 4  # Up to 250 lags
    acf = spatial_autocorrelation(x_final, max_lag=max_lag)
    lag_indices = np.arange(len(acf))
    lag_x = lag_indices * dx  # Convert to actual x distance

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        f"Spatial Autocorrelation: Initial Position Perturbation\n"
        f"({n_samples} pendulums, t = {t_sample}s, initial x ∈ [-2L, -L])",
        fontsize=13,
    )

    # Plot 1: Initial vs Final mapping
    ax1 = axes[0]
    ax1.plot(x_initial, x_final, ".", markersize=1.5, color="steelblue", alpha=0.5)
    ax1.set_xlabel("Initial x₀", fontsize=11)
    ax1.set_ylabel(f"Final x at t={t_sample}s", fontsize=11)
    ax1.set_title("Mapping: Initial → Final Position", fontsize=12)
    ax1.set_xlim(-2.1, -0.9)
    ax1.set_ylim(-2.2, 2.2)
    ax1.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spatial autocorrelation vs lag index
    ax2 = axes[1]
    ax2.plot(lag_x, acf, color="coral", lw=1.5)
    ax2.axhline(0, color="gray", lw=1, ls="-")
    ax2.axhline(0.05, color="red", lw=0.8, ls="--", alpha=0.5)
    ax2.axhline(-0.05, color="red", lw=0.8, ls="--", alpha=0.5)
    ax2.fill_between(lag_x, -0.05, 0.05, color="red", alpha=0.1)
    ax2.set_xlabel("Spatial lag Δx₀ (initial position difference)", fontsize=11)
    ax2.set_ylabel("Autocorrelation", fontsize=11)
    ax2.set_title("Spatial Autocorrelation Function", fontsize=12)
    ax2.set_xlim(0, lag_x[-1])
    ax2.set_ylim(-0.5, 1.05)
    ax2.grid(True, alpha=0.3)

    # Find decorrelation length
    below_half = np.where(acf < 0.5)[0]
    if len(below_half) > 0:
        half_corr_idx = below_half[0]
        half_corr_x = half_corr_idx * dx
        ax2.axvline(half_corr_x, color="purple", lw=1.5, ls=":",
                    label=f"τ₁/₂ = {half_corr_x:.4f}")
        ax2.legend(loc="upper right", fontsize=10)
        print(f"\nHalf-correlation length: Δx = {half_corr_x:.6f}")
        print(f"  (pendulums starting {half_corr_x:.6f} apart are 50% correlated)")

    # Find first zero crossing
    zero_crossings = np.where(np.diff(np.sign(acf)))[0]
    if len(zero_crossings) > 0:
        first_zero_idx = zero_crossings[0]
        first_zero_x = first_zero_idx * dx
        ax2.axvline(first_zero_x, color="green", lw=1.5, ls="--", alpha=0.7)
        print(f"First zero crossing: Δx = {first_zero_x:.6f}")

    # Plot 3: Zoomed autocorrelation (first 10% of lags)
    ax3 = axes[2]
    zoom_end = max_lag // 10
    ax3.plot(lag_x[:zoom_end], acf[:zoom_end], color="coral", lw=2)
    ax3.axhline(0, color="gray", lw=1, ls="-")
    ax3.axhline(0.5, color="purple", lw=0.8, ls=":", alpha=0.7)
    ax3.set_xlabel("Spatial lag Δx₀ (zoomed)", fontsize=11)
    ax3.set_ylabel("Autocorrelation", fontsize=11)
    ax3.set_title("Zoomed: Initial Decorrelation Region", fontsize=12)
    ax3.set_xlim(0, lag_x[zoom_end - 1])
    ax3.set_ylim(-0.2, 1.05)
    ax3.grid(True, alpha=0.3)

    # Annotate the rapid decay
    if len(below_half) > 0 and half_corr_idx < zoom_end:
        ax3.annotate(
            f"50% correlation\nat Δx = {half_corr_x:.5f}",
            xy=(half_corr_x, 0.5),
            xytext=(half_corr_x + lag_x[zoom_end] * 0.2, 0.65),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="purple"),
            color="purple",
        )

    plt.tight_layout()
    plt.savefig("spatial_autocorrelation.png", dpi=150, transparent=True)
    print("\nSaved figure to spatial_autocorrelation.png")

    # Save data
    data = np.column_stack([lag_x, acf])
    np.savetxt(
        "spatial_autocorrelation.csv",
        data,
        delimiter=",",
        header="lag_x,autocorrelation",
        comments="",
    )
    print("Saved data to spatial_autocorrelation.csv")

    plt.show()


if __name__ == "__main__":
    main()
