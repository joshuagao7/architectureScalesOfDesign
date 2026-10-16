"""
Generate a panel of pendulum swing images.

Columns: Different starting position ranges (each with 25 pendulums)
  - Column 1: x from -2L to -1.5L
  - Column 2: x from -1.5L to -1L
  - ... and so on to +2L

Rows: Time snapshots at 1s, 2s, ..., 10s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from double_pendulum_sim import (
    inverse_kinematics_end_effector,
    simulate_double_pendulum,
    generate_trajectories,
)


def generate_trajectories_for_range(x_start, x_end, n_pendulums, L1, L2, m1, m2, g, dt, t_max, elbow="up"):
    """
    Generate trajectories for pendulums with initial x positions from x_start to x_end.
    """
    x_positions = np.linspace(x_start, x_end, n_pendulums)
    target_y = 0.0
    
    trajectories = []
    valid_x = []
    
    for x in x_positions:
        try:
            theta1_0, theta2_0 = inverse_kinematics_end_effector(x, target_y, L1, L2, elbow=elbow)
            traj = simulate_double_pendulum(theta1_0, theta2_0, L1, L2, m1, m2, g, dt, t_max)
            trajectories.append(traj)
            valid_x.append(x)
        except ValueError:
            # Position unreachable, skip
            continue
    
    return valid_x, trajectories


def draw_pendulums_at_time(ax, trajectories, dt, sample_time, L1, L2, full_trail=True):
    """
    Draw all pendulums at a specific time snapshot with full historical trail.
    Greyscale version - rods only, no masses.
    """
    frame = int(sample_time / dt)
    
    n = len(trajectories)
    # Greyscale gradient - darker to lighter
    greys = np.linspace(0.15, 0.6, n)
    
    for i, (x1, y1, x2, y2) in enumerate(trajectories):
        idx = min(frame, len(x1) - 1)
        grey_val = greys[i]
        color = (grey_val, grey_val, grey_val)
        
        # Draw full historical trail for end effector (from t=0 to current time)
        if full_trail and idx > 0:
            trail_x = x2[0:idx+1]
            trail_y = y2[0:idx+1]
            # Draw trail as a single continuous line for visibility
            ax.plot(trail_x, trail_y, color=color, lw=0.8, alpha=0.6)
        
        # Draw first rod (pivot to mass 1) - just a line
        ax.plot([0, x1[idx]], [0, y1[idx]], 
                color=color, lw=1.5, alpha=0.9)
        
        # Draw second rod (mass 1 to mass 2) - just a line
        ax.plot([x1[idx], x2[idx]], [y1[idx], y2[idx]], 
                color=color, lw=1.5, alpha=0.9)


def draw_endpoints_at_time(ax, trajectories, dt, sample_time, marker_size=6):
    """
    Draw only the end-effector positions at a specific time snapshot.
    Greyscale version - dots only.
    """
    frame = int(sample_time / dt)
    
    n = len(trajectories)
    greys = np.linspace(0.15, 0.6, n)
    
    for i, (_, _, x2, y2) in enumerate(trajectories):
        idx = min(frame, len(x2) - 1)
        grey_val = greys[i]
        color = (grey_val, grey_val, grey_val)
        ax.plot(
            x2[idx],
            y2[idx],
            marker="o",
            markersize=marker_size,
            markeredgewidth=0.0,
            color=color,
            alpha=0.7,
            linestyle="None",
        )


def create_pendulum_panel(
    n_pendulums=25,
    L1=1.0,
    L2=1.0,
    m1=1.0,
    m2=1.0,
    g=9.81,
    dt=0.005,
    t_max=12.0,
    output_path="pendulum_panel.png",
    dpi=300,
    column_ranges=None,
    time_snapshots=None,
    figure_scale=4.0,
    render_mode="rods_trails",
    endpoint_size=6,
):
    """
    Create the full panel of pendulum visualizations.
    """
    # Default column ranges (each 0.5L wide) - right 4 columns only
    if column_ranges is None:
        L = L1  # Assuming L1 == L2
        column_ranges = [
            (0.0, 0.5 * L),
            (0.5 * L, 1.0 * L),
            (1.0 * L, 1.5 * L),
            (1.5 * L, 2.0 * L),
        ]
    
    # Default time snapshots for rows: 1s to 10s
    if time_snapshots is None:
        time_snapshots = np.arange(1.0, 10.5, 1.0)  # 1.0 to 10.0 seconds
    
    n_cols = len(column_ranges)
    n_rows = len(time_snapshots)
    
    print(f"Generating panel: {n_rows} rows × {n_cols} columns")
    print(f"Each cell: {n_pendulums} pendulums")
    print(f"Time snapshots: {time_snapshots}")
    print(f"Column ranges: {column_ranges}")
    
    # Pre-compute all trajectories for each column
    print("\nGenerating trajectories...")
    all_trajectories = []
    for col_idx, (x_start, x_end) in enumerate(column_ranges):
        print(f"  Column {col_idx + 1}: x ∈ [{x_start:.2f}, {x_end:.2f}]")
        valid_x, trajs = generate_trajectories_for_range(
            x_start, x_end, n_pendulums, L1, L2, m1, m2, g, dt, t_max
        )
        all_trajectories.append(trajs)
        print(f"    Generated {len(trajs)} pendulums")
    
    # Create figure - no borders, seamless grid
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figure_scale * n_cols, figure_scale * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    reach = L1 + L2
    
    print("\nDrawing panels...")
    for row_idx, t in enumerate(time_snapshots):
        for col_idx, (x_start, x_end) in enumerate(column_ranges):
            ax = axes[row_idx, col_idx]
            
            # Set up axes - transparent background
            ax.set_facecolor('none')
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-reach * 1.05, reach * 1.05)
            ax.set_ylim(-reach * 1.05, reach * 1.05)
            
            # Draw pendulums with full trails
            trajs = all_trajectories[col_idx]
            if trajs:
                if render_mode == "endpoints":
                    draw_endpoints_at_time(ax, trajs, dt, t, marker_size=endpoint_size)
                else:
                    draw_pendulums_at_time(
                        ax,
                        trajs,
                        dt,
                        t,
                        L1,
                        L2,
                        full_trail=(render_mode == "rods_trails"),
                    )
            
            # Remove all spines and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    # Remove spacing between subplots
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Save (SVG or PNG based on extension)
    is_svg = str(output_path).lower().endswith(".svg")
    if is_svg:
        print(f"\nSaving to {output_path} (SVG)...")
        plt.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
    else:
        print(f"\nSaving to {output_path} at {dpi} DPI...")
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
    plt.close()
    print("Done!")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pendulum panel visualization.")
    parser.add_argument("--n-pendulums", type=int, default=25,
                        help="Number of pendulums per cell (default: 25)")
    parser.add_argument("--output", default="pendulum_panel.png",
                        help="Output file path (default: pendulum_panel.png). Use .svg for vector output.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (default: 300)")
    parser.add_argument("--x-start", type=float, default=None,
                        help="Single column start x (units of L)")
    parser.add_argument("--x-end", type=float, default=None,
                        help="Single column end x (units of L)")
    parser.add_argument("--time-start", type=float, default=None,
                        help="Snapshot start time (seconds)")
    parser.add_argument("--time-end", type=float, default=None,
                        help="Snapshot end time (seconds)")
    parser.add_argument("--time-step", type=float, default=None,
                        help="Snapshot time step (seconds)")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Simulation max time (seconds)")
    parser.add_argument("--mode", choices=["rods_trails", "rods", "endpoints"],
                        default="rods_trails",
                        help="Render mode (default: rods_trails)")
    parser.add_argument("--endpoint-size", type=float, default=6.0,
                        help="Endpoint marker size (default: 6)")
    args = parser.parse_args()

    column_ranges = None
    if args.x_start is not None or args.x_end is not None:
        if args.x_start is None or args.x_end is None:
            raise SystemExit("Provide both --x-start and --x-end.")
        column_ranges = [(args.x_start, args.x_end)]

    time_snapshots = None
    if args.time_start is not None or args.time_end is not None or args.time_step is not None:
        if args.time_start is None or args.time_end is None or args.time_step is None:
            raise SystemExit("Provide --time-start, --time-end, and --time-step together.")
        time_snapshots = np.arange(args.time_start, args.time_end + 1e-9, args.time_step)
        time_snapshots = np.round(time_snapshots, 6)

    t_max = args.t_max
    if t_max is None and time_snapshots is not None:
        t_max = float(np.max(time_snapshots))
    if t_max is None:
        t_max = 12.0

    create_pendulum_panel(
        n_pendulums=args.n_pendulums,
        output_path=args.output,
        dpi=args.dpi,
        t_max=t_max,
        column_ranges=column_ranges,
        time_snapshots=time_snapshots,
        render_mode=args.mode,
        endpoint_size=args.endpoint_size,
    )
