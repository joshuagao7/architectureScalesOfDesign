"""
Sample endpoint positions from multiple double pendulum simulations.

This script runs N simulations with initial end-effector positions ranging
from fully extended left (x = -2L) to halfway extended left (x = -L) at y = 0,
then samples the x, y coordinates of both pendulum bobs at a specified time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from double_pendulum_sim import generate_trajectories


def sample_positions_at_time(
    trajectories, dt, sample_time
):
    """
    Extract (x1, y1, x2, y2) for each trajectory at the given sample_time.

    Returns:
        positions: np.ndarray of shape (N, 4) with columns [x1, y1, x2, y2]
    """
    frame = int(sample_time / dt)
    n = len(trajectories)
    positions = np.empty((n, 4), dtype=float)
    for i, (x1, y1, x2, y2) in enumerate(trajectories):
        # Clamp frame index in case sample_time exceeds simulation
        idx = min(frame, len(x1) - 1)
        positions[i] = [x1[idx], y1[idx], x2[idx], y2[idx]]
    return positions


def run_sampling(
    n_samples=100,
    L1=1.0,
    L2=1.0,
    m1=1.0,
    m2=1.0,
    g=9.81,
    dt=0.01,
    t_max=10.0,
    sample_time=10.0,
    elbow="up",
):
    """
    Generate trajectories for n_samples initial positions from x = -2L to x = -L
    at y = 0, and return the sampled positions at sample_time.

    Returns:
        x_initial: array of initial x positions
        trajectories: list of (x1, y1, x2, y2) arrays
        sampled_positions: (n_samples, 4) array of [x1, y1, x2, y2] at sample_time
    """
    L = L1  # Assuming L1 == L2 for the "one rod length" interpretation
    x_initial = np.linspace(-2 * L, -L, n_samples)
    target_y = 0.0

    trajectories = generate_trajectories(
        x_initial, target_y, L1, L2, m1, m2, g, dt, t_max, elbow=elbow
    )

    sampled_positions = sample_positions_at_time(trajectories, dt, sample_time)

    return x_initial, trajectories, sampled_positions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample double pendulum positions at a given time."
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of initial positions."
    )
    parser.add_argument(
        "--sample-time", type=float, default=10.0, help="Time (s) at which to sample."
    )
    parser.add_argument(
        "--output", default=None, help="Optional CSV file to save sampled positions."
    )
    args = parser.parse_args()

    x_initial, trajectories, sampled = run_sampling(
        n_samples=args.n_samples,
        sample_time=args.sample_time,
        t_max=max(args.sample_time, 10.0),
    )

    print(f"Sampled {len(sampled)} pendulums at t = {args.sample_time}s")
    print(f"Initial x range: [{x_initial[0]:.3f}, {x_initial[-1]:.3f}]")
    print(f"Sample positions shape: {sampled.shape}")
    print("\nFirst 5 sampled positions (x1, y1, x2, y2):")
    for i in range(min(5, len(sampled))):
        print(f"  {i}: x_init={x_initial[i]:.3f} -> {sampled[i]}")

    if args.output:
        header = "x_initial,x1,y1,x2,y2"
        data = np.column_stack([x_initial, sampled])
        np.savetxt(args.output, data, delimiter=",", header=header, comments="")
        print(f"\nSaved to {args.output}")
