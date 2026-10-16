"""
Animate 100 double pendulums with initial x from -2L to -L at y=0.

Uses the sampler module to generate trajectories, then animates all of them
with trails and marks the sampled positions at t=10s.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from double_pendulum_sampler import run_sampling


def main():
    parser = argparse.ArgumentParser(
        description="Animate 100 double pendulums from x=-2L to x=-L at y=0."
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output file (e.g. sampler_animation.gif).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive window (useful when saving).",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=2.0,
        help="Seconds of trail to keep.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of pendulums to simulate.",
    )
    parser.add_argument(
        "--sample-time",
        type=float,
        default=10.0,
        help="Time (s) at which to mark sampled positions.",
    )
    args = parser.parse_args()

    # Physical parameters (matching sampler defaults)
    L1 = 1.0
    L2 = 1.0
    dt = 0.01
    t_max = max(args.sample_time + 2.0, 12.0)  # Run a bit past sample time

    print(f"Simulating {args.n_samples} pendulums...")
    x_initial, trajectories, sampled_positions = run_sampling(
        n_samples=args.n_samples,
        dt=dt,
        t_max=t_max,
        sample_time=args.sample_time,
    )
    print(f"Simulation complete. Animating...")

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")
    reach = L1 + L2
    ax.set_xlim(-reach * 1.2, reach * 1.2)
    ax.set_ylim(-reach * 1.2, reach * 1.2)
    ax.set_title(f"100 Double Pendulums: x₀ ∈ [-2L, -L] at y=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)

    # Color gradient from dark purple to yellow
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(trajectories)))

    # Create line objects for each pendulum (rods only, no trail for clarity with 100)
    lines = []
    for color in colors:
        line, = ax.plot([], [], lw=0.8, color=color, alpha=0.6)
        lines.append(line)

    # Trails for end effector (second bob)
    trails = []
    for color in colors:
        trail, = ax.plot([], [], lw=0.5, color=color, alpha=0.25)
        trails.append(trail)

    # Markers for sampled positions (will appear at sample_time)
    sample_frame = int(args.sample_time / dt)
    sample_scatter = ax.scatter([], [], s=10, c="red", zorder=5, label=f"t={args.sample_time}s")

    # Time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    trail_len = max(1, int(args.trail_seconds / dt))

    def init():
        for line, trail in zip(lines, trails):
            line.set_data([], [])
            trail.set_data([], [])
        sample_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return lines + trails + [sample_scatter, time_text]

    def update(frame):
        t = frame * dt
        time_text.set_text(f"t = {t:.2f}s")

        start = max(0, frame - trail_len + 1)
        for idx, (line, trail) in enumerate(zip(lines, trails)):
            x1, y1, x2, y2 = trajectories[idx]
            line.set_data([0.0, x1[frame], x2[frame]], [0.0, y1[frame], y2[frame]])
            trail.set_data(x2[start : frame + 1], y2[start : frame + 1])

        # Show sampled positions once we reach sample_time
        if frame >= sample_frame:
            # Plot end effector positions (x2, y2) at sample time
            sample_scatter.set_offsets(sampled_positions[:, 2:4])
        else:
            sample_scatter.set_offsets(np.empty((0, 2)))

        return lines + trails + [sample_scatter, time_text]

    frames = len(trajectories[0][0])
    interval_ms = int(dt * 1000)
    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=interval_ms, blit=True
    )

    if args.save:
        fps = max(1, int(round(1.0 / dt)))
        print(f"Saving to {args.save} (this may take a while)...")
        anim.save(args.save, writer="pillow", fps=fps)
        print(f"Saved animation to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
