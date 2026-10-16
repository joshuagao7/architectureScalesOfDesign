import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from double_pendulum_sim import generate_trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Animate multiple double pendulums from y=0 with varying x."
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output file (e.g. double_pendulum_trails.gif).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive window (useful when saving).",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=4.0,
        help="Seconds of trail to keep.",
    )
    args = parser.parse_args()
    # Physical parameters
    L1 = 1.0
    L2 = 1.0
    m1 = 1.0
    m2 = 1.0
    g = 9.81

    # Simulation timing
    dt = 0.01
    t_max = 20.0
    trail_seconds = args.trail_seconds

    # Set initial end-effector positions along y=0 with varying x.
    # Keep x within [abs(L1 - L2), L1 + L2] to be reachable.
    x_positions = np.linspace(-1.8, 1.8, 5)
    target_y = 0.0
    elbow = "up"  # or "down"

    trajectories = generate_trajectories(
        x_positions, target_y, L1, L2, m1, m2, g, dt, t_max, elbow=elbow
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")
    reach = L1 + L2
    ax.set_xlim(-reach * 1.1, reach * 1.1)
    ax.set_ylim(-reach * 1.1, reach * 1.1)
    ax.set_title("Double Pendulum: Multiple Initial x at y=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(trajectories)))
    lines = []
    trails = []
    for color in colors:
        line, = ax.plot([], [], marker="o", markersize=4, lw=1.4, color=color)
        trail, = ax.plot([], [], lw=1.0, color=color, alpha=0.35)
        lines.append(line)
        trails.append(trail)

    trail_len = max(1, int(trail_seconds / dt))

    def init():
        for line, trail in zip(lines, trails):
            line.set_data([], [])
            trail.set_data([], [])
        return lines + trails

    def update(frame):
        start = max(0, frame - trail_len + 1)
        for idx, (line, trail) in enumerate(zip(lines, trails)):
            x1, y1, x2, y2 = trajectories[idx]
            line.set_data([0.0, x1[frame], x2[frame]], [0.0, y1[frame], y2[frame]])
            trail.set_data(x2[start : frame + 1], y2[start : frame + 1])
        return lines + trails

    frames = len(trajectories[0][0])
    interval_ms = int(dt * 1000)
    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=interval_ms, blit=True
    )

    if args.save:
        fps = max(1, int(round(1.0 / dt)))
        anim.save(args.save, writer="pillow", fps=fps)
        print(f"Saved animation to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
