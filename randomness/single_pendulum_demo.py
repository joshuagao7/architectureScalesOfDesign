"""
Single Double Pendulum Demo for Classroom Presentation.

A clean, focused animation of one double pendulum for teaching purposes.
Run with different initial positions to build intuition before showing chaos.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from double_pendulum_sim import (
    inverse_kinematics_end_effector,
    simulate_double_pendulum,
)


def main():
    parser = argparse.ArgumentParser(
        description="Animate a single double pendulum for demonstration."
    )
    parser.add_argument(
        "--x0",
        type=float,
        default=-1.5,
        help="Initial x position of end effector (default: -1.5).",
    )
    parser.add_argument(
        "--y0",
        type=float,
        default=0.0,
        help="Initial y position of end effector (default: 0.0).",
    )
    parser.add_argument(
        "--theta1",
        type=float,
        default=None,
        help="Initial angle of first rod in degrees (overrides x0/y0).",
    )
    parser.add_argument(
        "--theta2",
        type=float,
        default=None,
        help="Initial angle of second rod in degrees (overrides x0/y0).",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=3.0,
        help="Seconds of trail to display.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Total simulation time in seconds.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save animation to file (e.g., single_pendulum.gif).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show interactive window.",
    )
    args = parser.parse_args()

    # Physical parameters
    L1 = 1.0
    L2 = 1.0
    m1 = 1.0
    m2 = 1.0
    g = 9.81
    dt = 0.02  # Slightly larger dt for smoother presentation

    # Determine initial angles
    if args.theta1 is not None and args.theta2 is not None:
        theta1_0 = np.radians(args.theta1)
        theta2_0 = np.radians(args.theta2)
        mode = f"θ₁={args.theta1}°, θ₂={args.theta2}°"
    else:
        theta1_0, theta2_0 = inverse_kinematics_end_effector(
            args.x0, args.y0, L1, L2, elbow="up"
        )
        mode = f"x₀={args.x0}, y₀={args.y0}"

    print(f"Initial configuration: {mode}")
    print(f"  θ₁ = {np.degrees(theta1_0):.1f}°")
    print(f"  θ₂ = {np.degrees(theta2_0):.1f}°")

    # Run simulation
    x1, y1, x2, y2 = simulate_double_pendulum(
        theta1_0, theta2_0, L1, L2, m1, m2, g, dt, args.duration
    )

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")
    reach = L1 + L2
    ax.set_xlim(-reach * 1.15, reach * 1.15)
    ax.set_ylim(-reach * 1.15, reach * 1.15)
    ax.set_title("Double Pendulum", fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    # Reference lines
    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4)

    # Pivot point
    ax.plot(0, 0, "ko", markersize=10, zorder=10)

    # Pendulum rods and masses
    rod1, = ax.plot([], [], "o-", color="steelblue", lw=3, markersize=12,
                    markerfacecolor="royalblue", markeredgecolor="black")
    rod2, = ax.plot([], [], "o-", color="coral", lw=3, markersize=12,
                    markerfacecolor="orangered", markeredgecolor="black")

    # Trail for end effector
    trail, = ax.plot([], [], lw=1.5, color="orangered", alpha=0.4)

    # Time display
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                        fontsize=12, family="monospace")

    # Info text
    info_text = ax.text(0.98, 0.98, f"L₁ = L₂ = {L1}m\nm₁ = m₂ = {m1}kg\ng = {g} m/s²",
                        transform=ax.transAxes, va="top", ha="right",
                        fontsize=10, family="monospace", alpha=0.7)

    trail_len = max(1, int(args.trail_seconds / dt))

    def init():
        rod1.set_data([], [])
        rod2.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return rod1, rod2, trail, time_text

    def update(frame):
        t = frame * dt
        time_text.set_text(f"t = {t:.2f} s")

        # First rod: pivot to mass 1
        rod1.set_data([0, x1[frame]], [0, y1[frame]])

        # Second rod: mass 1 to mass 2
        rod2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])

        # Trail
        start = max(0, frame - trail_len + 1)
        trail.set_data(x2[start:frame + 1], y2[start:frame + 1])

        return rod1, rod2, trail, time_text

    frames = len(x1)
    interval_ms = int(dt * 1000)
    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=interval_ms, blit=True
    )

    if args.save:
        fps = max(1, int(round(1.0 / dt)))
        print(f"Saving to {args.save}...")
        anim.save(args.save, writer="pillow", fps=fps)
        print(f"Saved!")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
