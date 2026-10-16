"""
Plot x-coordinate time series and autocorrelation for a single double pendulum.

Shows the trajectory starting from x = -2L (fully extended left) and computes
the autocorrelation function to reveal loss of predictability over time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from double_pendulum_sim import (
    inverse_kinematics_end_effector,
    simulate_double_pendulum,
)


def autocorrelation(x, max_lag=None):
    """
    Compute the autocorrelation of signal x for lags 0 to max_lag.
    
    Returns normalized autocorrelation where lag 0 = 1.0.
    """
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    
    # Remove mean for proper autocorrelation
    x_centered = x - np.mean(x)
    
    # Variance for normalization
    var = np.sum(x_centered ** 2)
    
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.sum(x_centered[:-lag] * x_centered[lag:]) / var
    
    return acf


def main():
    # Physical parameters
    L1 = 1.0
    L2 = 1.0
    m1 = 1.0
    m2 = 1.0
    g = 9.81

    # Simulation parameters
    dt = 0.01
    t_max = 100.0

    # Initial position: fully extended left (x = -2L, y = 0)
    x0 = -2 * L1  # = -2.0
    y0 = 0.0

    theta1_0, theta2_0 = inverse_kinematics_end_effector(x0, y0, L1, L2, elbow="up")

    print(f"Initial position: x = {x0}, y = {y0}")
    print(f"Initial angles: θ₁ = {np.degrees(theta1_0):.1f}°, θ₂ = {np.degrees(theta2_0):.1f}°")
    print(f"Simulating for {t_max}s...")

    # Run simulation
    x1, y1, x2, y2 = simulate_double_pendulum(
        theta1_0, theta2_0, L1, L2, m1, m2, g, dt, t_max
    )

    # Time array
    t = np.arange(len(x2)) * dt

    # Compute autocorrelation
    max_lag_seconds = 50.0  # Look at lags up to 50 seconds
    max_lag_samples = int(max_lag_seconds / dt)
    acf = autocorrelation(x2, max_lag=max_lag_samples)
    lag_times = np.arange(len(acf)) * dt

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
    fig.suptitle(
        f"Double Pendulum: x-coordinate from x₀ = -2L\n"
        f"(L₁ = L₂ = {L1}m, m₁ = m₂ = {m1}kg, g = {g} m/s²)",
        fontsize=13,
    )

    # Top plot: x-coordinate time series
    ax1.plot(t, x2, color="steelblue", lw=0.8)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("x₂ position", fontsize=11)
    ax1.set_title("End-Effector x-Coordinate Over Time", fontsize=12)
    ax1.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(-2.2, 2.2)
    ax1.grid(True, alpha=0.3)

    # Add some statistics
    ax1.text(
        0.98, 0.95,
        f"mean = {x2.mean():.3f}\nstd = {x2.std():.3f}",
        transform=ax1.transAxes,
        ha="right", va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none"),
    )

    # Bottom plot: Autocorrelation
    ax2.plot(lag_times, acf, color="coral", lw=1.5)
    ax2.axhline(0, color="gray", lw=1, ls="-")
    ax2.axhline(0.05, color="red", lw=0.8, ls="--", alpha=0.5, label="±0.05 threshold")
    ax2.axhline(-0.05, color="red", lw=0.8, ls="--", alpha=0.5)
    ax2.fill_between(lag_times, -0.05, 0.05, color="red", alpha=0.1)
    ax2.set_xlabel("Lag (s)", fontsize=11)
    ax2.set_ylabel("Autocorrelation", fontsize=11)
    ax2.set_title("Autocorrelation Function", fontsize=12)
    ax2.set_xlim(0, max_lag_seconds)
    ax2.set_ylim(-0.5, 1.05)
    ax2.grid(True, alpha=0.3)

    # Find first zero crossing (decorrelation time)
    zero_crossings = np.where(np.diff(np.sign(acf)))[0]
    if len(zero_crossings) > 0:
        first_zero = zero_crossings[0] * dt
        ax2.axvline(first_zero, color="green", lw=1.5, ls=":", 
                    label=f"First zero crossing: {first_zero:.2f}s")
        ax2.legend(loc="upper right", fontsize=9)
    else:
        ax2.legend(loc="upper right", fontsize=9)

    # Find when autocorrelation drops below 0.5 (half-correlation time)
    below_half = np.where(acf < 0.5)[0]
    if len(below_half) > 0:
        half_corr_time = below_half[0] * dt
        ax2.annotate(
            f"τ₁/₂ = {half_corr_time:.2f}s",
            xy=(half_corr_time, 0.5),
            xytext=(half_corr_time + 1, 0.6),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="purple"),
            color="purple",
        )

    plt.tight_layout()
    plt.savefig("pendulum_autocorrelation.png", dpi=150, transparent=True)
    print("\nSaved figure to pendulum_autocorrelation.png")
    plt.show()


if __name__ == "__main__":
    main()
