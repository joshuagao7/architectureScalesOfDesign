"""
Progressive rejection of triangles outside the human figure.

Goal: Achieve a 3:1 ratio of triangles inside vs outside the human figure.
Method: Probabilistic rejection using pendulum-derived random numbers.
Output: Scrollable image sequence showing the rejection process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from PIL import Image
import os

# Paths to data
SCATTER_PATH = os.path.join(os.path.dirname(__file__), "scatter_coordinates.csv")
PENDULUM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "randomness", "spatial_perturbation", "sample_initial_positions.csv"
)
HUMAN_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "Male-athletic-human-body-silhouette-SVG-Graphics-80227886-2-580x387.jpg"
)
HUMAN_SVG_TRACE_PATH = os.path.join(os.path.dirname(__file__), "human_silhouette_trace.svg")
REJECTION_FRAMES_DIR = os.path.join(os.path.dirname(__file__), "rejection_frames")

# Canvas dimensions - large format for high resolution
# Larger area + sparser density to keep ~80% blank space.
CANVAS_SCALE = 3
CANVAS_WIDTH = 33 * CANVAS_SCALE  # 3x original (11 * 3) then scaled
CANVAS_HEIGHT = 51 * CANVAS_SCALE  # 3x original (17 * 3) then scaled
TRIANGLE_SCALE = 4.0
TRIANGLE_FRACTION = 0.8  # doubled again (was 0.4) for more triangles
SMALL_CANVAS_SCALE = 1
SMALL_TRIANGLE_FRACTION = 0.25


def rgba_to_hex(rgba):
    """Convert RGBA (0-1 floats) to hex and alpha."""
    r, g, b, a = rgba
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}", a


def write_svg_polygons(vertices_list, color_values, output_path, width, height, cmap, tight_bounds=True):
    """Write an SVG with one polygon per triangle for Figma editing."""
    if vertices_list and tight_bounds:
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


def otsu_threshold(gray):
    """Compute Otsu's threshold for a grayscale image array."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 0

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = i

    return threshold


def write_svg_outline(vertices, output_path, width, height):
    """Write a single silhouette polygon to SVG."""
    points = " ".join(f"{x:.3f},{y:.3f}" for x, y in vertices)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.3f}" height="{height:.3f}" '
        f'viewBox="0 0 {width:.3f} {height:.3f}">',
        f'  <polygon points="{points}" fill="#000000" />',
        "</svg>",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def get_human_outline_from_image(
    image_path,
    canvas_width,
    canvas_height,
    target_height_ratio=0.78,
    base_y_ratio=0.10,
    width_ratio=0.90,
    save_svg_path=None,
):
    """
    Trace a silhouette from a raster image and map it to the canvas.
    Returns a Path object for point-in-polygon testing and the outline vertices.
    """
    img = Image.open(image_path).convert("L")
    gray = np.array(img)

    threshold = otsu_threshold(gray)
    mask = gray <= threshold

    if mask.any() and (~mask).any():
        if gray[mask].mean() > gray[~mask].mean():
            mask = ~mask

    fig, ax = plt.subplots()
    contour = ax.contour(mask.astype(float), levels=[0.5])
    plt.close(fig)

    segments = contour.allsegs[0]
    if not segments:
        raise ValueError("No silhouette contour found in image.")

    def polygon_area(segment):
        x = segment[:, 0]
        y = segment[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    outline = max(segments, key=polygon_area)

    h, w = mask.shape
    x = outline[:, 0]
    y = (h - 1) - outline[:, 1]  # flip y-axis for plot coordinate space

    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())
    outline_width = max_x - min_x
    outline_height = max_y - min_y

    if outline_width <= 0 or outline_height <= 0:
        raise ValueError("Invalid silhouette bounds after tracing.")

    target_height = canvas_height * target_height_ratio
    target_width = canvas_width * width_ratio
    scale = min(target_height / outline_height, target_width / outline_width)

    x = (x - min_x) * scale
    y = (y - min_y) * scale

    x_offset = (canvas_width - outline_width * scale) / 2
    y_offset = canvas_height * base_y_ratio

    x += x_offset
    y += y_offset

    vertices = np.column_stack([x, y])
    vertices_closed = np.vstack([vertices, vertices[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1) + [Path.CLOSEPOLY]
    path = Path(vertices_closed, codes)

    if save_svg_path:
        write_svg_outline(vertices, save_svg_path, canvas_width, canvas_height)

    return path, vertices


def get_human_outline_path(canvas_width, canvas_height):
    """
    Create a realistic Vitruvian Man pose using Leonardo's proportions.
    Arms extended horizontally, legs spread ~30 degrees.
    
    Based on Leonardo da Vinci's ideal proportions:
    - Height = arm span
    - Head = 1/8 of height
    - Shoulder width = 1/4 of height
    - Leg = 1/2 of height
    
    Returns a Path object for point-in-polygon testing.
    """
    # Use 80% of canvas height for figure
    H = canvas_height * 0.78  # Total height
    cx = canvas_width / 2
    base_y = canvas_height * 0.10
    
    # Leonardo's proportions (as fractions of height H)
    head_h = H / 8
    head_w = head_h * 0.75
    neck_h = H * 0.03
    neck_w = head_w * 0.45
    
    shoulder_w = H / 4  # Full shoulder span
    chest_w = shoulder_w * 0.85
    waist_w = shoulder_w * 0.65
    hip_w = shoulder_w * 0.75
    
    # Arm span = height (Leonardo's rule)
    arm_span = H
    arm_length = (arm_span - shoulder_w) / 2
    upper_arm = arm_length * 0.47
    forearm = arm_length * 0.38
    hand = arm_length * 0.15
    arm_thick = H * 0.028
    
    # Leg proportions
    leg_length = H * 0.47
    thigh = leg_length * 0.55
    calf = leg_length * 0.45
    thigh_w = H * 0.065
    calf_w = H * 0.045
    ankle_w = H * 0.022
    foot_l = H * 0.15
    foot_w = H * 0.035
    
    # Leg spread angle (~25 degrees from vertical)
    leg_angle = 25 * np.pi / 180
    leg_spread_x = np.sin(leg_angle) * leg_length * 0.5
    
    # Key Y positions (from bottom)
    foot_y = base_y
    ankle_y = foot_y + H * 0.04
    knee_y = foot_y + calf
    crotch_y = foot_y + leg_length
    waist_y = crotch_y + H * 0.12
    chest_y = waist_y + H * 0.15
    shoulder_y = chest_y + H * 0.08
    neck_bottom_y = shoulder_y + H * 0.02
    neck_top_y = neck_bottom_y + neck_h
    head_top_y = neck_top_y + head_h
    
    arm_y = shoulder_y - H * 0.01  # Arms slightly below shoulder line
    
    vertices = []
    
    # === LEFT LEG (spread outward) ===
    lf_x = cx - leg_spread_x - foot_l * 0.3  # Left foot center
    la_x = cx - leg_spread_x * 0.9           # Left ankle
    lk_x = cx - leg_spread_x * 0.5           # Left knee
    lt_x = cx - thigh_w * 0.3                # Left thigh top
    
    # Left foot outer
    vertices.append((lf_x - foot_w, foot_y))
    vertices.append((lf_x - foot_w * 0.8, ankle_y))
    # Left calf outer
    vertices.append((la_x - ankle_w, ankle_y + H * 0.02))
    vertices.append((lk_x - calf_w * 0.9, knee_y - H * 0.03))
    vertices.append((lk_x - calf_w, knee_y))
    # Left thigh outer
    vertices.append((lk_x - thigh_w * 0.7, knee_y + H * 0.02))
    vertices.append((lt_x - thigh_w, crotch_y - H * 0.05))
    vertices.append((cx - hip_w / 2, crotch_y))
    
    # === LEFT TORSO ===
    vertices.append((cx - hip_w / 2 * 0.95, crotch_y + H * 0.03))
    vertices.append((cx - waist_w / 2, waist_y))
    vertices.append((cx - chest_w / 2, chest_y))
    vertices.append((cx - shoulder_w / 2, shoulder_y))
    
    # === LEFT ARM (horizontal) ===
    la_start = cx - shoulder_w / 2
    la_elbow = la_start - upper_arm
    la_wrist = la_elbow - forearm
    la_hand = la_wrist - hand
    
    # Armpit to fingertips (top edge)
    vertices.append((la_start - H * 0.01, arm_y + arm_thick))
    vertices.append((la_elbow, arm_y + arm_thick * 0.95))
    vertices.append((la_wrist, arm_y + arm_thick * 0.7))
    vertices.append((la_hand, arm_y + arm_thick * 0.4))
    # Fingertip
    vertices.append((la_hand - hand * 0.15, arm_y))
    # Back to armpit (bottom edge)
    vertices.append((la_hand, arm_y - arm_thick * 0.4))
    vertices.append((la_wrist, arm_y - arm_thick * 0.7))
    vertices.append((la_elbow, arm_y - arm_thick * 0.95))
    vertices.append((la_start - H * 0.01, arm_y - arm_thick))
    
    # === NECK AND HEAD ===
    vertices.append((cx - neck_w / 2, neck_bottom_y))
    vertices.append((cx - neck_w / 2 * 0.9, neck_top_y))
    # Head (oval shape)
    vertices.append((cx - head_w / 2, neck_top_y + head_h * 0.1))
    vertices.append((cx - head_w / 2 * 1.05, neck_top_y + head_h * 0.4))
    vertices.append((cx - head_w / 2 * 0.95, neck_top_y + head_h * 0.7))
    vertices.append((cx - head_w / 2 * 0.7, head_top_y - head_h * 0.05))
    vertices.append((cx, head_top_y))
    vertices.append((cx + head_w / 2 * 0.7, head_top_y - head_h * 0.05))
    vertices.append((cx + head_w / 2 * 0.95, neck_top_y + head_h * 0.7))
    vertices.append((cx + head_w / 2 * 1.05, neck_top_y + head_h * 0.4))
    vertices.append((cx + head_w / 2, neck_top_y + head_h * 0.1))
    vertices.append((cx + neck_w / 2 * 0.9, neck_top_y))
    vertices.append((cx + neck_w / 2, neck_bottom_y))
    
    # === RIGHT ARM (horizontal) ===
    ra_start = cx + shoulder_w / 2
    ra_elbow = ra_start + upper_arm
    ra_wrist = ra_elbow + forearm
    ra_hand = ra_wrist + hand
    
    # Armpit to fingertips (bottom edge first for clockwise)
    vertices.append((ra_start + H * 0.01, arm_y - arm_thick))
    vertices.append((ra_elbow, arm_y - arm_thick * 0.95))
    vertices.append((ra_wrist, arm_y - arm_thick * 0.7))
    vertices.append((ra_hand, arm_y - arm_thick * 0.4))
    # Fingertip
    vertices.append((ra_hand + hand * 0.15, arm_y))
    # Back (top edge)
    vertices.append((ra_hand, arm_y + arm_thick * 0.4))
    vertices.append((ra_wrist, arm_y + arm_thick * 0.7))
    vertices.append((ra_elbow, arm_y + arm_thick * 0.95))
    vertices.append((ra_start + H * 0.01, arm_y + arm_thick))
    
    # === RIGHT TORSO ===
    vertices.append((cx + shoulder_w / 2, shoulder_y))
    vertices.append((cx + chest_w / 2, chest_y))
    vertices.append((cx + waist_w / 2, waist_y))
    vertices.append((cx + hip_w / 2 * 0.95, crotch_y + H * 0.03))
    vertices.append((cx + hip_w / 2, crotch_y))
    
    # === RIGHT LEG (spread outward) ===
    rf_x = cx + leg_spread_x + foot_l * 0.3
    ra_x = cx + leg_spread_x * 0.9
    rk_x = cx + leg_spread_x * 0.5
    rt_x = cx + thigh_w * 0.3
    
    # Right thigh outer
    vertices.append((rt_x + thigh_w, crotch_y - H * 0.05))
    vertices.append((rk_x + thigh_w * 0.7, knee_y + H * 0.02))
    vertices.append((rk_x + calf_w, knee_y))
    # Right calf outer
    vertices.append((rk_x + calf_w * 0.9, knee_y - H * 0.03))
    vertices.append((ra_x + ankle_w, ankle_y + H * 0.02))
    vertices.append((rf_x + foot_w * 0.8, ankle_y))
    # Right foot outer
    vertices.append((rf_x + foot_w, foot_y))
    
    # === BOTTOM (between feet) ===
    # Right foot inner
    vertices.append((rf_x - foot_w * 0.3, foot_y))
    # Inner right leg going up
    vertices.append((ra_x - ankle_w * 0.5, ankle_y))
    vertices.append((rk_x - calf_w * 0.3, knee_y))
    vertices.append((rt_x + thigh_w * 0.2, crotch_y - H * 0.08))
    # Crotch
    vertices.append((cx, crotch_y - H * 0.02))
    # Inner left leg going down
    vertices.append((lt_x - thigh_w * 0.2, crotch_y - H * 0.08))
    vertices.append((lk_x + calf_w * 0.3, knee_y))
    vertices.append((la_x + ankle_w * 0.5, ankle_y))
    # Left foot inner
    vertices.append((lf_x + foot_w * 0.3, foot_y))
    
    # Close the path
    vertices.append(vertices[0])
    
    # Create Path
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    path = Path(vertices, codes)
    
    return path, vertices[:-1]  # Return path and vertices (without closing duplicate)


def create_isosceles_triangles(target_area=0.5):
    """
    Create 10 isosceles triangles with roughly equal areas.
    More oblong shapes for an athletic look.
    Larger base area for the bigger canvas.
    """
    triangles = []
    # More oblong ratios: favor wide bases and shorter heights
    # Example ratio: base/height = 10/4 = 2.5
    ratios = np.linspace(2.0, 4.5, 10)
    for ratio in ratios:
        height = np.sqrt(2 * target_area / ratio)
        base = ratio * height
        triangles.append((base, height))
    return triangles


def get_distance_scale(cx, cy, canvas_width, canvas_height):
    """
    Calculate size scale based on distance from figure center.
    Triangles closer to center are larger, farther ones are smaller.
    Returns scale factor between 0.3 and 1.0
    """
    # Figure center
    fig_cx = canvas_width / 2
    fig_cy = canvas_height * 0.55  # Roughly center of human figure
    
    # Distance from center (normalized)
    dx = (cx - fig_cx) / (canvas_width / 2)
    dy = (cy - fig_cy) / (canvas_height / 2)
    dist = np.sqrt(dx**2 + dy**2)
    
    # Scale: 1.0 at center, 0.3 at edges
    scale = max(0.3, 1.0 - 0.7 * min(dist, 1.0))
    return scale


def get_triangle_fit_score(vertices, human_path):
    """
    Calculate how well a triangle fits inside the human figure.
    Returns number of vertices inside (0, 1, 2, or 3).
    """
    inside = human_path.contains_points(vertices)
    return np.sum(inside)


def get_triangle_vertices(base, height, center_x, center_y, angle_deg):
    """Get vertices of an isosceles triangle."""
    centroid_y_offset = height / 3
    vertices = np.array([
        [0, height - centroid_y_offset],
        [-base / 2, -centroid_y_offset],
        [base / 2, -centroid_y_offset],
    ])
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = vertices @ rotation_matrix.T
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    return rotated


def map_to_bucket(value, min_val=-2, max_val=2, n_buckets=10):
    """Map a value to a bucket index."""
    value = np.clip(value, min_val, max_val)
    normalized = (value - min_val) / (max_val - min_val)
    bucket = int(normalized * n_buckets)
    return min(bucket, n_buckets - 1)


def map_to_angle(value, min_val=-2, max_val=2):
    """Map a value to [0, 360] degrees."""
    value = np.clip(value, min_val, max_val)
    normalized = (value - min_val) / (max_val - min_val)
    return normalized * 360


def map_to_uniform(value, min_val=-2, max_val=2):
    """Map pendulum value to [0, 1] for probability comparison."""
    value = np.clip(value, min_val, max_val)
    return (value - min_val) / (max_val - min_val)


def save_triangles_svg(triangles, output_path, canvas_width, canvas_height):
    """Save triangles as plasma-gradient SVG for editing."""
    cmap = plt.cm.plasma
    vertices_list = [t["vertices"] for t in triangles]
    color_values = [0.1 + 0.8 * (t["bucket"] / 9) for t in triangles]
    write_svg_polygons(
        vertices_list,
        color_values,
        output_path,
        canvas_width,
        canvas_height,
        cmap,
        tight_bounds=False,
    )


def main():
    # Load data
    scatter_data = np.loadtxt(SCATTER_PATH, delimiter=",", skiprows=1)
    centers_x = scatter_data[:, 0]
    centers_y = scatter_data[:, 1]
    n_points_total = len(centers_x)
    n_points = max(1, int(n_points_total * TRIANGLE_FRACTION))
    centers_x = centers_x[-n_points:] * CANVAS_SCALE
    centers_y = centers_y[-n_points:] * CANVAS_SCALE
    
    pendulum_data = np.loadtxt(PENDULUM_PATH, delimiter=",", skiprows=1)
    x_final = pendulum_data[:, 1]
    
    print(f"Loaded {n_points} of {n_points_total} triangle centers")
    
    # Generate shape and rotation values
    shape_values = x_final[0::4]
    rotation_values = x_final[1::4]
    rejection_values = x_final[2::4]  # For rejection decisions
    
    # Extend if needed
    def extend_array(arr, n):
        if len(arr) < n:
            return np.tile(arr, (n // len(arr) + 1))[:n]
        return arr[:n]
    
    shape_values = extend_array(shape_values, n_points)
    rotation_values = extend_array(rotation_values, n_points)
    # We need many rejection values for 1000 iterations
    rejection_values = extend_array(rejection_values, n_points * 2)
    
    # Create triangle presets
    triangle_presets = create_isosceles_triangles(target_area=0.16)  # doubled (was 0.08) for 2x area per triangle
    
    # Get human outline traced from the provided image
    human_path, human_vertices = get_human_outline_from_image(
        HUMAN_IMAGE_PATH,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
        target_height_ratio=0.78,
        base_y_ratio=0.10,
        width_ratio=0.90,
        save_svg_path=HUMAN_SVG_TRACE_PATH,
    )
    
    # Determine which triangles are inside the human figure
    points = np.column_stack([centers_x, centers_y])
    inside_mask = human_path.contains_points(points)
    
    n_inside = np.sum(inside_mask)
    n_outside = n_points - n_inside
    print(f"Triangles inside human: {n_inside}")
    print(f"Triangles outside human: {n_outside}")
    print(f"Initial ratio (outside/inside): {n_outside/n_inside:.3f}")
    print(f"Target ratio: {1/3:.3f}")
    
    # Build triangle data with fit scores and distance-based sizing
    triangles_data = []
    for i in range(n_points):
        bucket = map_to_bucket(shape_values[i])
        base, height = triangle_presets[bucket]
        
        # Scale triangle size based on distance from center
        # Triangles farther from center are smaller
        dist_scale = get_distance_scale(centers_x[i], centers_y[i], CANVAS_WIDTH, CANVAS_HEIGHT)
        scaled_base = base * dist_scale * TRIANGLE_SCALE
        scaled_height = height * dist_scale * TRIANGLE_SCALE
        
        angle = map_to_angle(rotation_values[i])
        vertices = get_triangle_vertices(scaled_base, scaled_height, centers_x[i], centers_y[i], angle)
        fit_score = get_triangle_fit_score(vertices, human_path)
        triangles_data.append({
            'vertices': vertices,
            'center': (centers_x[i], centers_y[i]),
            'bucket': bucket,
            'inside': inside_mask[i],  # Center is inside
            'fit_score': fit_score,     # 0-3 vertices inside
            'fully_contained': fit_score == 3,
            'active': True,
        })

    print("\nSaving plasma-gradient SVG of random triangles...")
    save_triangles_svg(triangles_data, "random_triangles.svg", CANVAS_WIDTH, CANVAS_HEIGHT)
    print("Saved random triangles to random_triangles.svg")

    # Also save the old small-area design (pre-rejection)
    small_canvas_width = 33 * SMALL_CANVAS_SCALE
    small_canvas_height = 51 * SMALL_CANVAS_SCALE
    small_n_points = max(1, int(n_points_total * SMALL_TRIANGLE_FRACTION))
    small_centers_x = scatter_data[-small_n_points:, 0] * SMALL_CANVAS_SCALE
    small_centers_y = scatter_data[-small_n_points:, 1] * SMALL_CANVAS_SCALE

    small_shape_values = x_final[0::4][:small_n_points]
    small_rotation_values = x_final[1::4][:small_n_points]
    small_shape_values = extend_array(small_shape_values, small_n_points)
    small_rotation_values = extend_array(small_rotation_values, small_n_points)

    small_vertices_list = []
    small_colors = []
    for i in range(small_n_points):
        bucket = map_to_bucket(small_shape_values[i])
        base, height = triangle_presets[bucket]
        dist_scale = get_distance_scale(
            small_centers_x[i], small_centers_y[i], small_canvas_width, small_canvas_height
        )
        scaled_base = base * dist_scale * TRIANGLE_SCALE
        scaled_height = height * dist_scale * TRIANGLE_SCALE
        angle = map_to_angle(small_rotation_values[i])
        vertices = get_triangle_vertices(
            scaled_base, scaled_height, small_centers_x[i], small_centers_y[i], angle
        )
        small_vertices_list.append(vertices)
        small_colors.append(0.1 + 0.8 * (bucket / 9))

    write_svg_polygons(
        small_vertices_list,
        small_colors,
        "random_triangles_small.svg",
        small_canvas_width,
        small_canvas_height,
        plt.cm.plasma,
    )
    print("Saved old small-area SVG to random_triangles_small.svg")
    
    # Count fit scores
    n_fully_contained = sum(1 for t in triangles_data if t['fully_contained'])
    n_partial = sum(1 for t in triangles_data if 0 < t['fit_score'] < 3)
    n_fully_outside = sum(1 for t in triangles_data if t['fit_score'] == 0)
    print(f"\nTriangle containment:")
    print(f"  Fully contained (3/3 vertices): {n_fully_contained}")
    print(f"  Partially inside (1-2 vertices): {n_partial}")
    print(f"  Fully outside (0/3 vertices): {n_fully_outside}")
    
    # Rejection process with fit-based reward/punishment
    # Goal: reject 7/8 of all triangles, strongly favoring fully contained triangles
    target_rejections = int(n_points * 7 / 8)
    n_snapshots = 12
    save_every_n_rejections = target_rejections // (n_snapshots - 1)
    
    # Track states for visualization
    states = []
    
    # Group triangles by containment level
    # Fully outside (0 vertices): very high rejection weight
    # Partial (1-2 vertices): moderate rejection weight  
    # Fully contained (3 vertices): very low rejection weight (protected)
    fully_outside_indices = [i for i, t in enumerate(triangles_data) if t['fit_score'] == 0]
    partial_indices = [i for i, t in enumerate(triangles_data) if 0 < t['fit_score'] < 3]
    fully_contained_indices = [i for i, t in enumerate(triangles_data) if t['fit_score'] == 3]
    
    # Rejection weights (higher = more likely to be rejected)
    weight_fully_outside = 50.0   # Very likely to reject
    weight_partial = 5.0          # Moderate
    weight_fully_contained = 0.5  # Protected - rarely reject
    
    rejection_idx = 0
    total_rejections = 0
    last_saved_rejection = -save_every_n_rejections
    
    print(f"\nTarget: reject {target_rejections} triangles (7/8 of {n_points})")
    print(f"Rejection weights: outside={weight_fully_outside}, partial={weight_partial}, contained={weight_fully_contained}")
    print(f"Rendering {n_snapshots} snapshots...")
    
    # Helper to count active triangles by category
    def count_active():
        n_out = sum(1 for i in fully_outside_indices if triangles_data[i]['active'])
        n_part = sum(1 for i in partial_indices if triangles_data[i]['active'])
        n_cont = sum(1 for i in fully_contained_indices if triangles_data[i]['active'])
        return n_out, n_part, n_cont
    
    # Save initial state
    n_out, n_part, n_cont = count_active()
    states.append({
        'rejections': 0,
        'n_outside': n_out,
        'n_partial': n_part,
        'n_contained': n_cont,
        'triangles': [{'vertices': t['vertices'], 'bucket': t['bucket'], 'active': t['active']} for t in triangles_data],
    })
    print(f"  Rejections 0: outside={n_out}, partial={n_part}, contained={n_cont}")
    
    max_iterations = target_rejections * 10
    iteration = 0
    
    while total_rejections < target_rejections and iteration < max_iterations:
        iteration += 1
        
        # Get active counts
        n_out, n_part, n_cont = count_active()
        total_active = n_out + n_part + n_cont
        
        if total_active == 0:
            break
        
        # Calculate weighted probabilities for each category
        weighted_out = weight_fully_outside * n_out
        weighted_part = weight_partial * n_part
        weighted_cont = weight_fully_contained * n_cont
        total_weight = weighted_out + weighted_part + weighted_cont
        
        if total_weight == 0:
            break
        
        p_reject_outside = weighted_out / total_weight
        p_reject_partial = weighted_part / total_weight
        # p_reject_contained = weighted_cont / total_weight (remainder)
        
        # Get random values
        rand_reject = map_to_uniform(rejection_values[rejection_idx % len(rejection_values)])
        rejection_idx += 1
        rand_select = map_to_uniform(rejection_values[rejection_idx % len(rejection_values)])
        rejection_idx += 1
        rand_which = map_to_uniform(rejection_values[rejection_idx % len(rejection_values)])
        rejection_idx += 1
        
        # Base rejection probability
        remaining = target_rejections - total_rejections
        base_prob = min(0.9, remaining / max(1, total_active))
        
        if rand_reject < base_prob:
            # Select which category to reject from based on weights
            if rand_select < p_reject_outside:
                # Reject from fully outside
                active_list = [i for i in fully_outside_indices if triangles_data[i]['active']]
                if active_list:
                    idx = active_list[int(rand_which * len(active_list))]
                    triangles_data[idx]['active'] = False
                    total_rejections += 1
            elif rand_select < p_reject_outside + p_reject_partial:
                # Reject from partial
                active_list = [i for i in partial_indices if triangles_data[i]['active']]
                if active_list:
                    idx = active_list[int(rand_which * len(active_list))]
                    triangles_data[idx]['active'] = False
                    total_rejections += 1
            else:
                # Reject from fully contained (rare)
                active_list = [i for i in fully_contained_indices if triangles_data[i]['active']]
                if active_list:
                    idx = active_list[int(rand_which * len(active_list))]
                    triangles_data[idx]['active'] = False
                    total_rejections += 1
        
        # Save state at intervals
        if total_rejections - last_saved_rejection >= save_every_n_rejections:
            n_out, n_part, n_cont = count_active()
            states.append({
                'rejections': total_rejections,
                'n_outside': n_out,
                'n_partial': n_part,
                'n_contained': n_cont,
                'triangles': [{'vertices': t['vertices'], 'bucket': t['bucket'], 'active': t['active']} for t in triangles_data],
            })
            print(f"  Rejections {total_rejections}: outside={n_out}, partial={n_part}, contained={n_cont}")
            last_saved_rejection = total_rejections
    
    # Save final state if not already saved
    if total_rejections > last_saved_rejection:
        n_out, n_part, n_cont = count_active()
        states.append({
            'rejections': total_rejections,
            'n_outside': n_out,
            'n_partial': n_part,
            'n_contained': n_cont,
            'triangles': [{'vertices': t['vertices'], 'bucket': t['bucket'], 'active': t['active']} for t in triangles_data],
        })
    
    # Final stats
    n_out, n_part, n_cont = count_active()
    final_total = n_out + n_part + n_cont
    print(f"\nFinal: total={final_total}")
    print(f"  Fully outside: {n_out}")
    print(f"  Partial (1-2 vertices): {n_part}")
    print(f"  Fully contained: {n_cont}")
    print(f"Rejected {total_rejections} triangles ({100*total_rejections/n_points:.1f}%)")
    
    n_states = len(states)
    cmap = plt.cm.plasma
    
    print(f"\nGenerating {n_states} high-resolution frames...")
    
    # Also save individual frames
    print("\nSaving individual frames...")
    os.makedirs(REJECTION_FRAMES_DIR, exist_ok=True)
    
    for state_idx, state in enumerate(states):
        # Large figure for high resolution
        fig_single, ax = plt.subplots(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT))
        ax.set_xlim(0, CANVAS_WIDTH)
        ax.set_ylim(0, CANVAS_HEIGHT)
        ax.set_aspect('equal')
        
        # Remove all axes, ticks, labels - just triangles
        ax.axis('off')
        fig_single.patch.set_alpha(0)  # Transparent background
        
        # Draw triangles (no outlines, just fill)
        patches = []
        colors = []
        
        for t in state['triangles']:
            if t['active']:
                triangle = MplPolygon(t['vertices'], closed=True)
                patches.append(triangle)
                colors.append(0.1 + 0.8 * (t['bucket'] / 9))
        
        if patches:
            collection = PatchCollection(
                patches,
                alpha=0.85,
                edgecolors='none',  # No outlines
                linewidths=0,
                cmap=cmap,
            )
            collection.set_array(np.array(colors))
            ax.add_collection(collection)
        
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(REJECTION_FRAMES_DIR, f"frame_{state_idx:03d}.svg"), format="svg",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig_single)
    
    print(f"Saved {n_states} individual frames to {REJECTION_FRAMES_DIR}")
    
    # Save human outline target as its own frame
    print("\nSaving target human outline...")
    fig_target, ax = plt.subplots(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT))
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(0, CANVAS_HEIGHT)
    ax.set_aspect('equal')
    ax.axis('off')
    fig_target.patch.set_alpha(0)  # Transparent background
    
    # Draw human outline only
    human_polygon = MplPolygon(
        human_vertices,
        fill=False,
        edgecolor='red',
        linewidth=4,
        linestyle='-',
        alpha=1.0,
    )
    ax.add_patch(human_polygon)
    
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(REJECTION_FRAMES_DIR, "frame_target.svg"), format="svg",
                bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_target)
    print(f"Saved target outline to {os.path.join(REJECTION_FRAMES_DIR, 'frame_target.svg')}")


if __name__ == "__main__":
    main()
