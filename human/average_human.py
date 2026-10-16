"""
Draw an average human silhouette using ANSUR II anthropometric data.

ANSUR II (2012) measured ~6,000 US Army personnel.
Data source: https://www.openlab.psu.edu/ansur2/

Key measurements used (in mm, then converted to normalized proportions):
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Circle, Arc
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# ANSUR II Average Measurements (in mm)
# Source: ANSUR II Technical Report, 2012
# Using weighted average of male and female data

ANSUR_II_DATA = {
    # Male averages (n=4,082)
    "male": {
        "stature": 1756,              # Total height
        "shoulder_breadth": 465,       # Biacromial breadth
        "hip_breadth": 360,            # Bitrochanteric breadth  
        "chest_breadth": 328,          # Chest breadth
        "waist_breadth": 296,          # Waist breadth (estimated from circumference)
        "head_length": 196,            # Head length
        "head_breadth": 155,           # Head breadth
        "neck_circumference": 394,     # Neck circumference
        "shoulder_height": 1438,       # Shoulder height (standing)
        "crotch_height": 830,          # Crotch height (leg length)
        "knee_height": 546,            # Knee height
        "ankle_height": 97,            # Ankle height
        "arm_length": 787,             # Arm length (shoulder to wrist)
        "hand_length": 193,            # Hand length
        "foot_length": 270,            # Foot length
        "foot_breadth": 101,           # Foot breadth
    },
    # Female averages (n=1,986)
    "female": {
        "stature": 1628,
        "shoulder_breadth": 395,
        "hip_breadth": 369,
        "chest_breadth": 285,
        "waist_breadth": 262,
        "head_length": 185,
        "head_breadth": 145,
        "neck_circumference": 337,
        "shoulder_height": 1336,
        "crotch_height": 757,
        "knee_height": 498,
        "ankle_height": 89,
        "arm_length": 713,
        "hand_length": 175,
        "foot_length": 245,
        "foot_breadth": 89,
    },
}


def get_average_measurements():
    """Compute average of male and female measurements."""
    avg = {}
    for key in ANSUR_II_DATA["male"]:
        avg[key] = (ANSUR_II_DATA["male"][key] + ANSUR_II_DATA["female"][key]) / 2
    return avg


def draw_human_silhouette(ax, measurements, x_offset=0, color="black", alpha=0.8):
    """
    Draw a stylized human silhouette using anthropometric measurements.
    
    All measurements are normalized to stature = 1.0, then scaled to fit canvas.
    """
    m = measurements
    stature = m["stature"]
    
    # Normalize all measurements to stature
    def norm(val):
        return val / stature
    
    # Key proportions (as fraction of height)
    head_height = norm(m["head_length"])
    head_width = norm(m["head_breadth"])
    neck_width = norm(m["neck_circumference"]) / np.pi  # Diameter from circumference
    shoulder_width = norm(m["shoulder_breadth"])
    chest_width = norm(m["chest_breadth"])
    waist_width = norm(m["waist_breadth"])
    hip_width = norm(m["hip_breadth"])
    
    shoulder_y = norm(m["shoulder_height"])
    crotch_y = norm(m["crotch_height"])
    knee_y = norm(m["knee_height"])
    ankle_y = norm(m["ankle_height"])
    
    arm_length = norm(m["arm_length"])
    hand_length = norm(m["hand_length"])
    foot_length = norm(m["foot_length"])
    foot_width = norm(m["foot_breadth"])
    
    # Calculate derived positions
    head_center_y = 1.0 - head_height / 2
    neck_y = 1.0 - head_height
    chest_y = shoulder_y - 0.1
    waist_y = (shoulder_y + crotch_y) / 2
    
    # Limb widths (estimated proportions)
    upper_arm_width = shoulder_width * 0.22
    forearm_width = shoulder_width * 0.18
    hand_width = shoulder_width * 0.20
    thigh_width = hip_width * 0.35
    calf_width = hip_width * 0.25
    ankle_width = hip_width * 0.15
    
    patches = []
    
    # HEAD (ellipse)
    head = Ellipse(
        (x_offset, head_center_y),
        width=head_width,
        height=head_height,
        color=color,
        alpha=alpha,
    )
    patches.append(head)
    
    # NECK (rectangle approximated as polygon)
    neck_top = neck_y
    neck_bottom = shoulder_y + 0.02
    neck_verts = [
        (x_offset - neck_width/2, neck_top),
        (x_offset + neck_width/2, neck_top),
        (x_offset + neck_width/2, neck_bottom),
        (x_offset - neck_width/2, neck_bottom),
    ]
    neck = Polygon(neck_verts, color=color, alpha=alpha)
    patches.append(neck)
    
    # TORSO (polygon with smooth transitions)
    torso_verts = [
        # Left shoulder
        (x_offset - shoulder_width/2, shoulder_y),
        # Left armpit
        (x_offset - chest_width/2, chest_y),
        # Left waist
        (x_offset - waist_width/2, waist_y),
        # Left hip
        (x_offset - hip_width/2, crotch_y + 0.05),
        # Crotch
        (x_offset, crotch_y),
        # Right hip
        (x_offset + hip_width/2, crotch_y + 0.05),
        # Right waist
        (x_offset + waist_width/2, waist_y),
        # Right armpit
        (x_offset + chest_width/2, chest_y),
        # Right shoulder
        (x_offset + shoulder_width/2, shoulder_y),
        # Neck right
        (x_offset + neck_width/2, shoulder_y + 0.02),
        # Neck left
        (x_offset - neck_width/2, shoulder_y + 0.02),
    ]
    torso = Polygon(torso_verts, color=color, alpha=alpha)
    patches.append(torso)
    
    # ARMS (left and right)
    for side in [-1, 1]:
        shoulder_x = x_offset + side * shoulder_width / 2
        elbow_y = shoulder_y - arm_length * 0.55
        wrist_y = shoulder_y - arm_length
        hand_bottom_y = wrist_y - hand_length
        
        # Upper arm
        upper_arm_verts = [
            (shoulder_x - upper_arm_width/2, shoulder_y),
            (shoulder_x + upper_arm_width/2, shoulder_y),
            (shoulder_x + side * forearm_width/2 + forearm_width/2, elbow_y),
            (shoulder_x + side * forearm_width/2 - forearm_width/2, elbow_y),
        ]
        upper_arm = Polygon(upper_arm_verts, color=color, alpha=alpha)
        patches.append(upper_arm)
        
        # Forearm
        forearm_verts = [
            (shoulder_x + side * forearm_width/2 - forearm_width/2, elbow_y),
            (shoulder_x + side * forearm_width/2 + forearm_width/2, elbow_y),
            (shoulder_x + side * hand_width/2 + hand_width/2 * 0.7, wrist_y),
            (shoulder_x + side * hand_width/2 - hand_width/2 * 0.7, wrist_y),
        ]
        forearm = Polygon(forearm_verts, color=color, alpha=alpha)
        patches.append(forearm)
        
        # Hand (ellipse)
        hand_x = shoulder_x + side * hand_width/2
        hand = Ellipse(
            (hand_x, wrist_y - hand_length/2),
            width=hand_width * 0.8,
            height=hand_length,
            color=color,
            alpha=alpha,
        )
        patches.append(hand)
    
    # LEGS (left and right)
    leg_spread = hip_width * 0.25
    for side in [-1, 1]:
        hip_x = x_offset + side * leg_spread
        
        # Thigh
        thigh_verts = [
            (hip_x - thigh_width/2, crotch_y),
            (hip_x + thigh_width/2, crotch_y),
            (hip_x + calf_width/2, knee_y),
            (hip_x - calf_width/2, knee_y),
        ]
        thigh = Polygon(thigh_verts, color=color, alpha=alpha)
        patches.append(thigh)
        
        # Calf
        calf_verts = [
            (hip_x - calf_width/2, knee_y),
            (hip_x + calf_width/2, knee_y),
            (hip_x + ankle_width/2, ankle_y),
            (hip_x - ankle_width/2, ankle_y),
        ]
        calf = Polygon(calf_verts, color=color, alpha=alpha)
        patches.append(calf)
        
        # Foot
        foot_verts = [
            (hip_x - ankle_width/2, ankle_y),
            (hip_x + ankle_width/2, ankle_y),
            (hip_x + foot_length * 0.3, 0),
            (hip_x - foot_length * 0.15, 0),
        ]
        foot = Polygon(foot_verts, color=color, alpha=alpha)
        patches.append(foot)
    
    for patch in patches:
        ax.add_patch(patch)
    
    return patches


def main():
    # Get average human measurements
    avg = get_average_measurements()
    male = ANSUR_II_DATA["male"]
    female = ANSUR_II_DATA["female"]
    
    print("ANSUR II Anthropometric Data Summary")
    print("=" * 50)
    print(f"{'Measurement':<25} {'Male':>10} {'Female':>10} {'Average':>10}")
    print("-" * 50)
    for key in avg:
        print(f"{key:<25} {male[key]:>10.0f} {female[key]:>10.0f} {avg[key]:>10.0f}")
    print("=" * 50)
    print("All measurements in mm")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    fig.suptitle(
        "Average Human Silhouette (ANSUR II Data, 2012)\n"
        "US Army Anthropometric Survey, n ≈ 6,000",
        fontsize=14,
    )
    
    titles = ["Male Average", "Combined Average", "Female Average"]
    datasets = [male, avg, female]
    colors = ["steelblue", "gray", "coral"]
    
    for ax, title, data, color in zip(axes, titles, datasets, colors):
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.05, 1.1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.3)
        
        draw_human_silhouette(ax, data, x_offset=0, color=color, alpha=0.85)
        
        # Add height annotation
        stature_m = data["stature"] / 1000
        ax.text(0.4, 0.5, f"{stature_m:.2f}m\n({stature_m*3.281:.1f}ft)", 
                fontsize=10, ha="left", va="center")
        
        # Add reference lines
        ax.axhline(data["shoulder_height"]/data["stature"], 
                   color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.axhline(data["crotch_height"]/data["stature"], 
                   color="gray", lw=0.5, ls=":", alpha=0.5)
        
        ax.set_ylabel("Proportion of height" if ax == axes[0] else "")
        ax.set_xlabel("Proportion of height")
    
    plt.tight_layout()
    plt.savefig("average_human.png", dpi=150, transparent=True)
    plt.savefig("average_human.svg", format="svg", transparent=True)
    print("\nSaved figures to average_human.png and average_human.svg")
    
    plt.show()


if __name__ == "__main__":
    main()
