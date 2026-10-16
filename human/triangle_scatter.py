"""
Scatter plot with isosceles triangles using chaotic pendulum data.

- Triangle centers: from scatter_coordinates.csv (pendulum-derived x,y pairs)
- Triangle shape: 10 preset isosceles triangles, selected by bucketing pendulum values
- Triangle rotation: mapped from pendulum values to 0-360 degrees
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os

# Path to the random data
SCATTER_PATH = os.path.join(os.path.dirname(__file__), "scatter_coordinates.csv")
PENDULUM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "randomness", "spatial_perturbation", "sample_initial_positions.csv"
)
TRIANGLE_SCALE = 4.0
TRIANGLE_FRACTION = 1.0
CANVAS_SCALE = 2
SMALL_TRIANGLE_FRACTION = 0.25
SMALL_CANVAS_SCALE = 1


def rgba_to_hex(rgba):
    """Convert RGBA (0-1 floats) to hex and alpha."""
    r, g, b, a = rgba
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}", a


def write_svg_polygons(vertices_list, color_values, output_path, width, height, cmap):
    """Write an SVG with one polygon per triangle for Figma editing."""
    if vertices_list:
        all_points = np.vstack(vertices_list)
        min_x = float(all_points[:, 0].min())
        max_x = float(all_points[:, 0].max())
        min_y = float(all_points[:, 1].min())
        max_y = float(all_points[:, 1].max())
        padding = 0.0
        view_min_x = min_x - padding
        view_min_y = min_y - padding
        svg_width = (max_x - min_x) + 2 * padding
        svg_height = (max_y - min_y) + 2 * padding
    else:
        view_min_x = 0.0
        view_min_y = 0.0
        svg_width = float(width)
        svg_height = float(height)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_width:.3f}" height="{svg_height:.3f}" '
        f'viewBox="{view_min_x:.3f} {view_min_y:.3f} {svg_width:.3f} {svg_height:.3f}">',
    ]

    for i, (vertices, color_value) in enumerate(zip(vertices_list, color_values)):
        hex_color, alpha = rgba_to_hex(cmap(color_value))
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in vertices)
        opacity_attr = f' fill-opacity="{alpha:.3f}"' if alpha < 1 else ""
        lines.append(
            f'  <polygon id="triangle_{i:04d}" points="{points}" '
            f'fill="{hex_color}"{opacity_attr} />'
        )

    lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def create_isosceles_triangles(target_area=0.15):
    """
    Create 10 isosceles triangles with roughly equal areas but different shapes.
    
    Area of isosceles triangle = (1/2) * base * height
    We vary the base-to-height ratio while keeping area constant.
    
    Returns list of (base, height) tuples, with apex at top.
    """
    triangles = []
    
    # More oblong shapes: favor wide bases and shorter heights
    # base/height ratios from 2.0 to 4.5 (example: 10/4 = 2.5)
    ratios = np.linspace(2.0, 4.5, 10)
    
    for ratio in ratios:
        # Area = 0.5 * base * height
        # base = ratio * height
        # Area = 0.5 * ratio * height^2
        # height = sqrt(2 * Area / ratio)
        height = np.sqrt(2 * target_area / ratio)
        base = ratio * height
        triangles.append((base, height))
    
    return triangles


def get_triangle_vertices(base, height, center_x, center_y, angle_deg):
    """
    Get vertices of an isosceles triangle centered at (center_x, center_y).
    
    Triangle has apex at top (before rotation).
    Angle is rotation in degrees.
    """
    # Vertices relative to center (apex up)
    # Center is at centroid (1/3 up from base)
    centroid_y_offset = height / 3  # Centroid is 1/3 from base
    
    vertices = np.array([
        [0, height - centroid_y_offset],           # Apex (top)
        [-base / 2, -centroid_y_offset],           # Bottom left
        [base / 2, -centroid_y_offset],            # Bottom right
    ])
    
    # Rotate
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated = vertices @ rotation_matrix.T
    
    # Translate to center
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    
    return rotated


def map_to_bucket(value, min_val=-2, max_val=2, n_buckets=10):
    """Map a value to a bucket index (0 to n_buckets-1)."""
    # Clamp value to range
    value = np.clip(value, min_val, max_val)
    # Normalize to [0, 1]
    normalized = (value - min_val) / (max_val - min_val)
    # Map to bucket
    bucket = int(normalized * n_buckets)
    # Clamp to valid range
    return min(bucket, n_buckets - 1)


def map_to_angle(value, min_val=-2, max_val=2):
    """Map a value from [-2, 2] to [0, 360] degrees."""
    value = np.clip(value, min_val, max_val)
    normalized = (value - min_val) / (max_val - min_val)
    return normalized * 360


def main():
    # Canvas dimensions
    canvas_width = 11 * CANVAS_SCALE
    canvas_height = 17 * CANVAS_SCALE
    
    # Load scatter coordinates (x, y centers)
    scatter_data = np.loadtxt(SCATTER_PATH, delimiter=",", skiprows=1)
    centers_x = scatter_data[:, 0]
    centers_y = scatter_data[:, 1]
    n_points_total = len(centers_x)
    n_points = max(1, int(n_points_total * TRIANGLE_FRACTION))
    centers_x = centers_x[-n_points:] * CANVAS_SCALE
    centers_y = centers_y[-n_points:] * CANVAS_SCALE
    
    print(f"Loaded {n_points} of {n_points_total} center points from scatter data")
    
    # Load raw pendulum data for shape selection and rotation
    pendulum_data = np.loadtxt(PENDULUM_PATH, delimiter=",", skiprows=1)
    x_final = pendulum_data[:, 1]  # Chaotic x values
    
    # We need 2 random values per point: one for shape, one for rotation
    # Use different parts of the sequence
    # Shape selection: use every 4th value starting at 0
    # Rotation: use every 4th value starting at 1
    shape_values = x_final[0::4][:n_points]
    rotation_values = x_final[1::4][:n_points]
    
    # If we don't have enough values, wrap around
    if len(shape_values) < n_points:
        shape_values = np.tile(shape_values, (n_points // len(shape_values) + 1))[:n_points]
    if len(rotation_values) < n_points:
        rotation_values = np.tile(rotation_values, (n_points // len(rotation_values) + 1))[:n_points]
    
    print(f"Shape values range: [{shape_values.min():.3f}, {shape_values.max():.3f}]")
    print(f"Rotation values range: [{rotation_values.min():.3f}, {rotation_values.max():.3f}]")
    
    # Create the 10 preset triangles
    triangle_presets = create_isosceles_triangles(target_area=0.08)
    
    print("\n10 Triangle Presets (base, height):")
    for i, (b, h) in enumerate(triangle_presets):
        area = 0.5 * b * h
        print(f"  Bucket {i}: base={b:.3f}, height={h:.3f}, area={area:.4f}")
    
    # Create figure for SVG export (no axes or labels)
    fig, ax = plt.subplots(figsize=(11, 17))
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)
    
    # Create triangles
    patches = []
    colors = []
    vertices_list = []
    
    # Plasma gradient (purple to yellow) for the 10 triangle types
    cmap = plt.cm.plasma
    
    for i in range(n_points):
        # Get center
        cx, cy = centers_x[i], centers_y[i]
        
        # Select triangle shape from bucket
        bucket = map_to_bucket(shape_values[i])
        base, height = triangle_presets[bucket]
        base *= TRIANGLE_SCALE
        height *= TRIANGLE_SCALE
        
        # Get rotation angle
        angle = map_to_angle(rotation_values[i])
        
        # Create triangle vertices
        vertices = get_triangle_vertices(base, height, cx, cy, angle)
        
        # Create polygon
        triangle = Polygon(vertices, closed=True)
        patches.append(triangle)
        colors.append(0.1 + 0.8 * (bucket / 9))  # Match animation range [0.1, 0.9]
        vertices_list.append(vertices)
    
    # Add all triangles as a collection
    collection = PatchCollection(
        patches,
        alpha=1.0,
        edgecolors="none",
        linewidths=0,
        cmap=cmap,
    )
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("triangle_scatter.png", dpi=150, transparent=True)
    write_svg_polygons(vertices_list, colors, "triangle_scatter.svg", canvas_width, canvas_height, cmap)
    print("\nSaved editable plasma-gradient SVG to triangle_scatter.svg")

    # Also save the old small-area design
    small_canvas_width = 11 * SMALL_CANVAS_SCALE
    small_canvas_height = 17 * SMALL_CANVAS_SCALE
    small_n_points = max(1, int(n_points_total * SMALL_TRIANGLE_FRACTION))
    small_centers_x = scatter_data[-small_n_points:, 0] * SMALL_CANVAS_SCALE
    small_centers_y = scatter_data[-small_n_points:, 1] * SMALL_CANVAS_SCALE

    small_shape_values = x_final[0::4][:small_n_points]
    small_rotation_values = x_final[1::4][:small_n_points]
    if len(small_shape_values) < small_n_points:
        small_shape_values = np.tile(
            small_shape_values, (small_n_points // len(small_shape_values) + 1)
        )[:small_n_points]
    if len(small_rotation_values) < small_n_points:
        small_rotation_values = np.tile(
            small_rotation_values, (small_n_points // len(small_rotation_values) + 1)
        )[:small_n_points]

    small_vertices_list = []
    small_colors = []
    for i in range(small_n_points):
        bucket = map_to_bucket(small_shape_values[i])
        base, height = triangle_presets[bucket]
        base *= TRIANGLE_SCALE
        height *= TRIANGLE_SCALE
        angle = map_to_angle(small_rotation_values[i])
        vertices = get_triangle_vertices(base, height, small_centers_x[i], small_centers_y[i], angle)
        small_vertices_list.append(vertices)
        small_colors.append(0.1 + 0.8 * (bucket / 9))

    write_svg_polygons(
        small_vertices_list,
        small_colors,
        "triangle_scatter_small.svg",
        small_canvas_width,
        small_canvas_height,
        cmap,
    )
    print("Saved old small-area SVG to triangle_scatter_small.svg")
    
    # Print statistics
    bucket_counts = np.bincount([map_to_bucket(v) for v in shape_values], minlength=10)
    print("\nTriangle type distribution:")
    for i, count in enumerate(bucket_counts):
        print(f"  Type {i}: {count} triangles ({100*count/n_points:.1f}%)")
    
    plt.show()


if __name__ == "__main__":
    main()
