"""
Gaussian-based triangle rejection.

For each horizontal band:
1. Find the width of the human figure at that height
2. Create a Gaussian centered at the figure's center with 2σ = figure width
3. Accept/reject triangles based on their x-coordinate's probability under this Gaussian

The result is a "soft" human shape that approximates but doesn't perfectly match the outline.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import os

# Paths to data
SCATTER_PATH = os.path.join(os.path.dirname(__file__), "scatter_coordinates.csv")
PENDULUM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "randomness", "spatial_perturbation", "sample_initial_positions.csv"
)

# Canvas dimensions
CANVAS_WIDTH = 11
CANVAS_HEIGHT = 17


def get_human_outline_path(canvas_width, canvas_height):
    """
    Create a realistic Vitruvian Man pose using Leonardo's proportions.
    Returns a Path object and the vertices.
    """
    H = canvas_height * 0.78
    cx = canvas_width / 2
    base_y = canvas_height * 0.10
    
    head_h = H / 8
    head_w = head_h * 0.75
    neck_h = H * 0.03
    neck_w = head_w * 0.45
    
    shoulder_w = H / 4
    chest_w = shoulder_w * 0.85
    waist_w = shoulder_w * 0.65
    hip_w = shoulder_w * 0.75
    
    arm_span = H
    arm_length = (arm_span - shoulder_w) / 2
    upper_arm = arm_length * 0.47
    forearm = arm_length * 0.38
    hand = arm_length * 0.15
    arm_thick = H * 0.028
    
    leg_length = H * 0.47
    thigh_w = H * 0.065
    calf_w = H * 0.045
    ankle_w = H * 0.022
    foot_l = H * 0.15
    foot_w = H * 0.035
    
    leg_angle = 25 * np.pi / 180
    leg_spread_x = np.sin(leg_angle) * leg_length * 0.5
    
    foot_y = base_y
    ankle_y = foot_y + H * 0.04
    calf = leg_length * 0.45
    knee_y = foot_y + calf
    crotch_y = foot_y + leg_length
    waist_y = crotch_y + H * 0.12
    chest_y = waist_y + H * 0.15
    shoulder_y = chest_y + H * 0.08
    neck_bottom_y = shoulder_y + H * 0.02
    neck_top_y = neck_bottom_y + neck_h
    head_top_y = neck_top_y + head_h
    
    arm_y = shoulder_y - H * 0.01
    
    vertices = []
    
    # Left leg
    lf_x = cx - leg_spread_x - foot_l * 0.3
    la_x = cx - leg_spread_x * 0.9
    lk_x = cx - leg_spread_x * 0.5
    lt_x = cx - thigh_w * 0.3
    
    vertices.append((lf_x - foot_w, foot_y))
    vertices.append((lf_x - foot_w * 0.8, ankle_y))
    vertices.append((la_x - ankle_w, ankle_y + H * 0.02))
    vertices.append((lk_x - calf_w * 0.9, knee_y - H * 0.03))
    vertices.append((lk_x - calf_w, knee_y))
    vertices.append((lk_x - thigh_w * 0.7, knee_y + H * 0.02))
    vertices.append((lt_x - thigh_w, crotch_y - H * 0.05))
    vertices.append((cx - hip_w / 2, crotch_y))
    
    # Left torso
    vertices.append((cx - hip_w / 2 * 0.95, crotch_y + H * 0.03))
    vertices.append((cx - waist_w / 2, waist_y))
    vertices.append((cx - chest_w / 2, chest_y))
    vertices.append((cx - shoulder_w / 2, shoulder_y))
    
    # Left arm
    la_start = cx - shoulder_w / 2
    la_elbow = la_start - upper_arm
    la_wrist = la_elbow - forearm
    la_hand = la_wrist - hand
    
    vertices.append((la_start - H * 0.01, arm_y + arm_thick))
    vertices.append((la_elbow, arm_y + arm_thick * 0.95))
    vertices.append((la_wrist, arm_y + arm_thick * 0.7))
    vertices.append((la_hand, arm_y + arm_thick * 0.4))
    vertices.append((la_hand - hand * 0.15, arm_y))
    vertices.append((la_hand, arm_y - arm_thick * 0.4))
    vertices.append((la_wrist, arm_y - arm_thick * 0.7))
    vertices.append((la_elbow, arm_y - arm_thick * 0.95))
    vertices.append((la_start - H * 0.01, arm_y - arm_thick))
    
    # Neck and head
    vertices.append((cx - neck_w / 2, neck_bottom_y))
    vertices.append((cx - neck_w / 2 * 0.9, neck_top_y))
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
    
    # Right arm
    ra_start = cx + shoulder_w / 2
    ra_elbow = ra_start + upper_arm
    ra_wrist = ra_elbow + forearm
    ra_hand = ra_wrist + hand
    
    vertices.append((ra_start + H * 0.01, arm_y - arm_thick))
    vertices.append((ra_elbow, arm_y - arm_thick * 0.95))
    vertices.append((ra_wrist, arm_y - arm_thick * 0.7))
    vertices.append((ra_hand, arm_y - arm_thick * 0.4))
    vertices.append((ra_hand + hand * 0.15, arm_y))
    vertices.append((ra_hand, arm_y + arm_thick * 0.4))
    vertices.append((ra_wrist, arm_y + arm_thick * 0.7))
    vertices.append((ra_elbow, arm_y + arm_thick * 0.95))
    vertices.append((ra_start + H * 0.01, arm_y + arm_thick))
    
    # Right torso
    vertices.append((cx + shoulder_w / 2, shoulder_y))
    vertices.append((cx + chest_w / 2, chest_y))
    vertices.append((cx + waist_w / 2, waist_y))
    vertices.append((cx + hip_w / 2 * 0.95, crotch_y + H * 0.03))
    vertices.append((cx + hip_w / 2, crotch_y))
    
    # Right leg
    rf_x = cx + leg_spread_x + foot_l * 0.3
    ra_x = cx + leg_spread_x * 0.9
    rk_x = cx + leg_spread_x * 0.5
    rt_x = cx + thigh_w * 0.3
    
    vertices.append((rt_x + thigh_w, crotch_y - H * 0.05))
    vertices.append((rk_x + thigh_w * 0.7, knee_y + H * 0.02))
    vertices.append((rk_x + calf_w, knee_y))
    vertices.append((rk_x + calf_w * 0.9, knee_y - H * 0.03))
    vertices.append((ra_x + ankle_w, ankle_y + H * 0.02))
    vertices.append((rf_x + foot_w * 0.8, ankle_y))
    vertices.append((rf_x + foot_w, foot_y))
    
    # Bottom between feet
    vertices.append((rf_x - foot_w * 0.3, foot_y))
    vertices.append((ra_x - ankle_w * 0.5, ankle_y))
    vertices.append((rk_x - calf_w * 0.3, knee_y))
    vertices.append((rt_x + thigh_w * 0.2, crotch_y - H * 0.08))
    vertices.append((cx, crotch_y - H * 0.02))
    vertices.append((lt_x - thigh_w * 0.2, crotch_y - H * 0.08))
    vertices.append((lk_x + calf_w * 0.3, knee_y))
    vertices.append((la_x + ankle_w * 0.5, ankle_y))
    vertices.append((lf_x + foot_w * 0.3, foot_y))
    
    vertices.append(vertices[0])
    
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    path = Path(vertices, codes)
    
    return path, vertices[:-1]


def get_figure_width_at_height(human_path, y, canvas_width, n_samples=200):
    """
    Find the width of the human figure at a given y-coordinate.
    Returns (center_x, width) or (None, 0) if figure doesn't exist at this height.
    """
    # Sample x-coordinates at this y level
    x_samples = np.linspace(0, canvas_width, n_samples)
    points = np.column_stack([x_samples, np.full(n_samples, y)])
    inside = human_path.contains_points(points)
    
    if not np.any(inside):
        return None, 0
    
    inside_x = x_samples[inside]
    left_edge = inside_x.min()
    right_edge = inside_x.max()
    center = (left_edge + right_edge) / 2
    width = right_edge - left_edge
    
    return center, width


def get_figure_height_at_x(human_path, x, canvas_height, n_samples=200):
    """
    Find the height extent of the human figure at a given x-coordinate.
    Returns (center_y, height) or (None, 0) if figure doesn't exist at this x.
    """
    # Sample y-coordinates at this x level
    y_samples = np.linspace(0, canvas_height, n_samples)
    points = np.column_stack([np.full(n_samples, x), y_samples])
    inside = human_path.contains_points(points)
    
    if not np.any(inside):
        return None, 0
    
    inside_y = y_samples[inside]
    bottom_edge = inside_y.min()
    top_edge = inside_y.max()
    center = (bottom_edge + top_edge) / 2
    height = top_edge - bottom_edge
    
    return center, height


def create_horizontal_bands(human_path, canvas_width, canvas_height, n_bands=30):
    """
    Create Gaussian parameters for each horizontal band.
    """
    bands = []
    band_height = canvas_height / n_bands
    
    for i in range(n_bands):
        y_min = i * band_height
        y_max = (i + 1) * band_height
        y_center = (y_min + y_max) / 2
        
        center, width = get_figure_width_at_height(human_path, y_center, canvas_width)
        
        if center is not None and width > 0:
            sigma = width / 2  # 2σ = width
            bands.append({
                'y_min': y_min,
                'y_max': y_max,
                'y_center': y_center,
                'center_x': center,
                'width': width,
                'sigma': sigma,
            })
        else:
            bands.append({
                'y_min': y_min,
                'y_max': y_max,
                'y_center': y_center,
                'center_x': canvas_width / 2,
                'width': 0,
                'sigma': canvas_width,
            })
    
    return bands


def create_vertical_bands(human_path, canvas_width, canvas_height, n_bands=25):
    """
    Create Gaussian parameters for each vertical band.
    """
    bands = []
    band_width = canvas_width / n_bands
    
    for i in range(n_bands):
        x_min = i * band_width
        x_max = (i + 1) * band_width
        x_center = (x_min + x_max) / 2
        
        center, height = get_figure_height_at_x(human_path, x_center, canvas_height)
        
        if center is not None and height > 0:
            sigma = height / 2  # 2σ = height
            bands.append({
                'x_min': x_min,
                'x_max': x_max,
                'x_center': x_center,
                'center_y': center,
                'height': height,
                'sigma': sigma,
            })
        else:
            bands.append({
                'x_min': x_min,
                'x_max': x_max,
                'x_center': x_center,
                'center_y': canvas_height / 2,
                'height': 0,
                'sigma': canvas_height,
            })
    
    return bands


def plot_bands_with_gaussians(human_path, human_vertices, h_bands, v_bands, canvas_width, canvas_height):
    """
    Create visualization of horizontal and vertical bands with Gaussian distributions overlaid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 22))
    
    cmap_h = plt.cm.coolwarm
    cmap_v = plt.cm.viridis
    
    # Top-left: Horizontal bands only
    ax1 = axes[0, 0]
    ax1.set_xlim(0, canvas_width)
    ax1.set_ylim(0, canvas_height)
    ax1.set_aspect('equal')
    
    for i, band in enumerate(h_bands):
        if band['width'] > 0:
            color = cmap_h(0.3 + 0.4 * (i / len(h_bands)))
            rect = Rectangle(
                (0, band['y_min']), canvas_width, band['y_max'] - band['y_min'],
                fill=True, facecolor=color, edgecolor='gray', linewidth=0.5, alpha=0.15,
            )
            ax1.add_patch(rect)
    ax1.axis("off")
    
    # Top-right: Horizontal Gaussians only
    ax2 = axes[0, 1]
    ax2.set_xlim(0, canvas_width)
    ax2.set_ylim(0, canvas_height)
    ax2.set_aspect('equal')
    
    x_range = np.linspace(0, canvas_width, 200)
    for i, band in enumerate(h_bands):
        if band['width'] > 0:
            z = (x_range - band['center_x']) / band['sigma']
            gauss = np.exp(-0.5 * z**2)
            gauss_norm = gauss / gauss.max()
            band_height = band['y_max'] - band['y_min']
            y_offset = band['y_min']
            color = cmap_h(0.3 + 0.4 * (i / len(h_bands)))
            ax2.fill_between(x_range, y_offset, y_offset + gauss_norm * band_height * 0.9,
                           color=color, alpha=0.4)
    ax2.axis("off")
    
    # Bottom-left: Vertical bands only
    ax3 = axes[1, 0]
    ax3.set_xlim(0, canvas_width)
    ax3.set_ylim(0, canvas_height)
    ax3.set_aspect('equal')
    
    for i, band in enumerate(v_bands):
        if band['height'] > 0:
            color = cmap_v(0.3 + 0.4 * (i / len(v_bands)))
            rect = Rectangle(
                (band['x_min'], 0), band['x_max'] - band['x_min'], canvas_height,
                fill=True, facecolor=color, edgecolor='gray', linewidth=0.5, alpha=0.15,
            )
            ax3.add_patch(rect)
    ax3.axis("off")
    
    # Bottom-right: Vertical Gaussians only
    ax4 = axes[1, 1]
    ax4.set_xlim(0, canvas_width)
    ax4.set_ylim(0, canvas_height)
    ax4.set_aspect('equal')
    
    y_range = np.linspace(0, canvas_height, 200)
    for i, band in enumerate(v_bands):
        if band['height'] > 0:
            z = (y_range - band['center_y']) / band['sigma']
            gauss = np.exp(-0.5 * z**2)
            gauss_norm = gauss / gauss.max()
            band_width = band['x_max'] - band['x_min']
            x_offset = band['x_min']
            color = cmap_v(0.3 + 0.4 * (i / len(v_bands)))
            ax4.fill_betweenx(y_range, x_offset, x_offset + gauss_norm * band_width * 0.9,
                            color=color, alpha=0.4)
    ax4.axis("off")
    
    plt.tight_layout(pad=0)
    plt.savefig("gaussian_bands.png", dpi=150, transparent=True)
    print("Saved gaussian_bands.png")
    plt.close()


def create_isosceles_triangles(target_area=0.06):
    """
    Create 10 isosceles triangles with roughly equal areas.
    More oblong shapes - ranging from very tall/narrow to moderately wide.
    """
    triangles = []
    # More oblong ratios: from very tall (0.2) to moderately wide (1.5)
    # This creates more athletic, dynamic shapes
    ratios = np.linspace(0.2, 1.5, 10)
    for ratio in ratios:
        # base/height = ratio, area = 0.5 * base * height
        # So: height = sqrt(2 * area / ratio), base = ratio * height
        height = np.sqrt(2 * target_area / ratio)
        base = ratio * height
        triangles.append((base, height))
    return triangles


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
    value = np.clip(value, min_val, max_val)
    normalized = (value - min_val) / (max_val - min_val)
    bucket = int(normalized * n_buckets)
    return min(bucket, n_buckets - 1)


def map_to_angle(value, min_val=-2, max_val=2):
    value = np.clip(value, min_val, max_val)
    normalized = (value - min_val) / (max_val - min_val)
    return normalized * 360


def map_to_uniform(value, min_val=-2, max_val=2):
    value = np.clip(value, min_val, max_val)
    return (value - min_val) / (max_val - min_val)


def get_horizontal_acceptance_prob(x, y, h_bands):
    """
    Get acceptance probability based on horizontal bands.
    """
    for band in h_bands:
        if band['y_min'] <= y < band['y_max']:
            if band['width'] == 0:
                return 0.05
            z = (x - band['center_x']) / band['sigma']
            return np.exp(-0.5 * z**2)
    return 0.05


def get_vertical_acceptance_prob(x, y, v_bands):
    """
    Get acceptance probability based on vertical bands.
    """
    for band in v_bands:
        if band['x_min'] <= x < band['x_max']:
            if band['height'] == 0:
                return 0.05
            z = (y - band['center_y']) / band['sigma']
            return np.exp(-0.5 * z**2)
    return 0.05


def get_triangle_fit_score(vertices, human_path):
    """
    Calculate how well a triangle fits inside the human figure.
    Returns a score from 0 to 1:
    - 1.0: All vertices inside (perfect fit)
    - 0.5: 2 vertices inside
    - 0.2: 1 vertex inside
    - 0.05: No vertices inside
    """
    inside = human_path.contains_points(vertices)
    n_inside = np.sum(inside)
    
    if n_inside == 3:
        return 1.0  # Perfect fit - strong reward
    elif n_inside == 2:
        return 0.5  # Mostly inside
    elif n_inside == 1:
        return 0.2  # Partially inside
    else:
        return 0.05  # Outside - low probability


def get_triangle_alignment_bonus(vertices, human_path, h_bands, v_bands):
    """
    Additional bonus for triangles that align well with the figure's shape.
    Rewards triangles that are:
    1. Fully contained within the figure
    2. Have their center deep inside (not on edge)
    """
    # Check containment
    inside = human_path.contains_points(vertices)
    n_inside = np.sum(inside)
    
    # Base containment score
    containment = n_inside / 3.0
    
    # Center point
    cx = np.mean(vertices[:, 0])
    cy = np.mean(vertices[:, 1])
    
    # Get Gaussian probability at center (measures how "deep" inside the figure)
    h_prob = get_horizontal_acceptance_prob(cx, cy, h_bands)
    v_prob = get_vertical_acceptance_prob(cx, cy, v_bands)
    depth = (h_prob * v_prob) ** 0.5  # Geometric mean
    
    # Combined score: containment * depth
    # Fully contained + deep inside = 1.0 * 1.0 = 1.0
    # Partially contained or on edge = lower
    score = containment * (0.5 + 0.5 * depth)
    
    return score


def main():
    # Load data
    scatter_data = np.loadtxt(SCATTER_PATH, delimiter=",", skiprows=1)
    centers_x = scatter_data[:, 0]
    centers_y = scatter_data[:, 1]
    n_points = len(centers_x)
    
    pendulum_data = np.loadtxt(PENDULUM_PATH, delimiter=",", skiprows=1)
    x_final = pendulum_data[:, 1]
    
    print(f"Loaded {n_points} triangle centers")
    
    # Generate shape and rotation values
    shape_values = x_final[0::4]
    rotation_values = x_final[1::4]
    rejection_values = x_final[2::4]
    
    def extend_array(arr, n):
        if len(arr) < n:
            return np.tile(arr, (n // len(arr) + 1))[:n]
        return arr[:n]
    
    shape_values = extend_array(shape_values, n_points)
    rotation_values = extend_array(rotation_values, n_points)
    rejection_values = extend_array(rejection_values, n_points * 3)
    
    # Create triangle presets
    triangle_presets = create_isosceles_triangles(target_area=0.08)
    
    # Get human outline
    human_path, human_vertices = get_human_outline_path(CANVAS_WIDTH, CANVAS_HEIGHT)
    
    # Create horizontal and vertical band Gaussians
    n_h_bands = 40
    n_v_bands = 25
    h_bands = create_horizontal_bands(human_path, CANVAS_WIDTH, CANVAS_HEIGHT, n_h_bands)
    v_bands = create_vertical_bands(human_path, CANVAS_WIDTH, CANVAS_HEIGHT, n_v_bands)
    
    print(f"Created {n_h_bands} horizontal bands and {n_v_bands} vertical bands")
    
    # Plot the bands visualization
    print("\nGenerating band visualization...")
    plot_bands_with_gaussians(human_path, human_vertices, h_bands, v_bands, CANVAS_WIDTH, CANVAS_HEIGHT)
    
    # Build triangle data
    triangles_data = []
    for i in range(n_points):
        bucket = map_to_bucket(shape_values[i])
        base, height = triangle_presets[bucket]
        angle = map_to_angle(rotation_values[i])
        vertices = get_triangle_vertices(base, height, centers_x[i], centers_y[i], angle)
        
        triangles_data.append({
            'vertices': vertices,
            'center': (centers_x[i], centers_y[i]),
            'bucket': bucket,
            'active': True,
        })
    
    # Rejection process using alternating horizontal and vertical Gaussian probabilities
    print("\nRunning alternating H/V Gaussian rejection process...")
    
    # Target: keep roughly 15-20% of triangles
    target_keep_fraction = 0.15
    
    # We'll alternate between horizontal and vertical passes
    states = []
    rejection_idx = 0
    
    # Save initial state
    n_active = sum(1 for t in triangles_data if t['active'])
    states.append({
        'pass': 0,
        'pass_type': 'initial',
        'n_active': n_active,
        'triangles': [{'vertices': t['vertices'], 'bucket': t['bucket'], 'active': t['active']} for t in triangles_data],
    })
    print(f"  Pass 0 (initial): {n_active} active triangles")
    
    # Pre-calculate fit scores for each triangle
    print("\nCalculating triangle fit scores...")
    for t in triangles_data:
        t['fit_score'] = get_triangle_fit_score(t['vertices'], human_path)
        t['alignment_bonus'] = get_triangle_alignment_bonus(
            t['vertices'], human_path, h_bands, v_bands
        )
    
    # Alternating rejection passes with fit reward
    n_passes = 12
    for pass_num in range(1, n_passes + 1):
        # Alternate between horizontal and vertical
        use_horizontal = (pass_num % 2 == 1)
        pass_type = "H" if use_horizontal else "V"
        
        for i, t in enumerate(triangles_data):
            if t['active']:
                cx, cy = t['center']
                
                # Get base acceptance probability based on direction
                if use_horizontal:
                    base_prob = get_horizontal_acceptance_prob(cx, cy, h_bands)
                else:
                    base_prob = get_vertical_acceptance_prob(cx, cy, v_bands)
                
                # Apply fit reward: triangles that fit neatly get a significant boost
                # fit_score ranges from 0.05 (outside) to 1.0 (all vertices inside)
                fit_bonus = t['fit_score']
                alignment_bonus = t['alignment_bonus']
                
                # Combined acceptance: base * (boosted by fit)
                # Triangles fully inside: prob * 1.5 (capped at 1.0)
                # Triangles partially inside: prob * 0.8
                # Triangles outside: prob * 0.3
                reward_multiplier = 0.3 + 1.2 * fit_bonus + 0.5 * alignment_bonus
                accept_prob = min(1.0, base_prob * reward_multiplier)
                
                # Get random value
                rand_val = map_to_uniform(rejection_values[rejection_idx % len(rejection_values)])
                rejection_idx += 1
                
                # Scale acceptance probability by pass (earlier passes are more lenient)
                pass_scale = 0.4 + 0.6 * (pass_num / n_passes)
                adjusted_prob = accept_prob ** pass_scale
                
                # Reject if random value exceeds adjusted probability
                if rand_val > adjusted_prob:
                    t['active'] = False
        
        n_active = sum(1 for t in triangles_data if t['active'])
        n_fully_inside = sum(1 for t in triangles_data if t['active'] and t['fit_score'] == 1.0)
        
        # Save state every pass
        states.append({
            'pass': pass_num,
            'pass_type': pass_type,
            'n_active': n_active,
            'n_fully_inside': n_fully_inside,
            'triangles': [{'vertices': t['vertices'], 'bucket': t['bucket'], 'active': t['active']} for t in triangles_data],
        })
        
        print(f"  Pass {pass_num} ({pass_type}): {n_active} active ({n_fully_inside} fully inside)")
        
        if n_active <= n_points * target_keep_fraction:
            break
    
    # Final stats
    final_active = sum(1 for t in triangles_data if t['active'])
    print(f"\nFinal: {final_active} triangles ({100*final_active/n_points:.1f}%)")
    
    # Generate frames
    print(f"\nGenerating {len(states)} frames...")
    os.makedirs("gaussian_frames", exist_ok=True)
    
    cmap = plt.cm.viridis
    
    for state_idx, state in enumerate(states):
        fig, ax = plt.subplots(figsize=(11, 17))
        ax.set_xlim(0, CANVAS_WIDTH)
        ax.set_ylim(0, CANVAS_HEIGHT)
        ax.set_aspect('equal')
        pass_type = state.get('pass_type', '')
        n_inside = state.get('n_fully_inside', '?')
        ax.set_title(f"Pass {state['pass']} ({pass_type}): {state['n_active']} triangles ({n_inside} fully inside)", fontsize=14)
        
        # Draw triangles
        patches = []
        colors = []
        
        for t in state['triangles']:
            if t['active']:
                triangle = MplPolygon(t['vertices'], closed=True)
                patches.append(triangle)
                colors.append(t['bucket'] / 9)
        
        if patches:
            collection = PatchCollection(
                patches,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.3,
                cmap=cmap,
            )
            collection.set_array(np.array(colors))
            ax.add_collection(collection)
        
        ax.set_xlabel("x (inches)")
        ax.set_ylabel("y (inches)")
        
        plt.tight_layout()
        plt.savefig(f"gaussian_frames/frame_{state_idx:03d}.png", dpi=150, transparent=True)
        plt.close(fig)
    
    print(f"Saved {len(states)} frames to gaussian_frames/")
    
    # Create scrollable composite
    n_states = len(states)
    fig_height = 17 * n_states
    fig, axes = plt.subplots(n_states, 1, figsize=(11, fig_height))
    
    if n_states == 1:
        axes = [axes]
    
    for ax_idx, state in enumerate(states):
        ax = axes[ax_idx]
        ax.set_xlim(0, CANVAS_WIDTH)
        ax.set_ylim(0, CANVAS_HEIGHT)
        ax.set_aspect('equal')
        pass_type = state.get('pass_type', '')
        n_inside = state.get('n_fully_inside', '?')
        ax.set_title(f"Pass {state['pass']} ({pass_type}): {state['n_active']} ({n_inside} inside)", fontsize=10)
        
        patches = []
        colors = []
        
        for t in state['triangles']:
            if t['active']:
                triangle = MplPolygon(t['vertices'], closed=True)
                patches.append(triangle)
                colors.append(t['bucket'] / 9)
        
        if patches:
            collection = PatchCollection(
                patches,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.2,
                cmap=cmap,
            )
            collection.set_array(np.array(colors))
            ax.add_collection(collection)
        
        ax.set_xlabel("x (inches)")
        ax.set_ylabel("y (inches)")
    
    plt.tight_layout()
    plt.savefig("gaussian_rejection.png", dpi=100, transparent=True)
    print("\nSaved composite to gaussian_rejection.png")


if __name__ == "__main__":
    main()
