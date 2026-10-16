"""
Generate a scatter field using random numbers from the double pendulum simulation.

Uses the chaotic x-coordinates to place points on an 11 x 17 canvas.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the random data
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "randomness", "spatial_perturbation", "sample_initial_positions.csv"
)


def load_random_values():
    """Load the chaotic x values from the pendulum simulation."""
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    x_final = data[:, 1]  # The chaotic x values at t=10s
    return x_final


def map_to_range(values, new_min, new_max):
    """Map values from their current range to [new_min, new_max]."""
    old_min, old_max = values.min(), values.max()
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def main():
    # Canvas dimensions - large format for high resolution
    canvas_width = 33  # 3x original
    canvas_height = 51  # 3x original

    # Load random values from pendulum simulation
    random_values = load_random_values()
    print(f"Loaded {len(random_values)} random values from pendulum simulation")
    print(f"Value range: [{random_values.min():.3f}, {random_values.max():.3f}]")

    # Use pairs of values as (x, y) coordinates
    # This gives us 500 points from 1000 values
    n_points = len(random_values) // 2
    x_coords = random_values[0::2]  # Even indices
    y_coords = random_values[1::2]  # Odd indices

    # Map to canvas dimensions
    x_canvas = map_to_range(x_coords, 0, canvas_width)
    y_canvas = map_to_range(y_coords, 0, canvas_height)

    print(f"\nGenerated {n_points} points on {canvas_width} x {canvas_height} canvas")

    # Create figure with exact canvas proportions
    fig, ax = plt.subplots(figsize=(11, 17))
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect("equal")

    # Plot scatter points
    ax.scatter(
        x_canvas, y_canvas,
        s=15,
        c="black",
        alpha=0.7,
        edgecolors="none",
    )

    # Keep only the points (no axes, labels, or grid)
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig("scatter_canvas.png", dpi=150, transparent=True)
    print("\nSaved figure to scatter_canvas.png")

    # Also save coordinates to CSV for further use
    coords = np.column_stack([x_canvas, y_canvas])
    np.savetxt(
        "scatter_coordinates.csv",
        coords,
        delimiter=",",
        header="x,y",
        comments="",
    )
    print("Saved coordinates to scatter_coordinates.csv")


if __name__ == "__main__":
    main()
