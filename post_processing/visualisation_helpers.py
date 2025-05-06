import numpy as np
import os 
import imageio.v2 as imageio_v2

"""VISUALISATION HELPERS: 
1. COLOURS: Defines the colours for body part sections 
2. Links Dictionaries: Define connecting joints using SMPL and pose prediction parameters
3. Calculate Axis Limits: Provides axis boundaries across animation frames so all axes have equal scales for 3D proportions 
4. Plot Skeleton Frame: Sets up a 3D plot with proper label, draws a floor grid beneath the skeleton and plots joints as black dots
5. Create Gif: Takes individual frames saved to iutput directory and combines into animated gif

"""
COLORS = {
    "torso": "green",
    "left_leg": "red",
    "right_leg": "blue",
    "left_arm": "orange",
    "right_arm": "purple",
}

torso_links = [
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
]

left_leg_links = [
    (0, 1),
    (1, 4),
    (4, 7),
    (7, 10),
]

right_leg_links = [
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
] 

left_arm_links = [
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
]  

right_arm_links = [
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
]

def calculate_axis_limits(joints_list):
    # Reshape and get min/max values
    all_joints = joints_list.reshape(-1, 3)
    mins = np.min(all_joints, axis=0)  # [x_min, y_min, z_min]
    maxs = np.max(all_joints, axis=0)  # [x_max, y_max, z_max]
    
    # Calculate ranges for the margin 
    margin = 0.1
    ranges = (maxs - mins) * (1 + 2 * margin)  # Add a margin to both sides for clearer visuals 
    max_range = max(ranges)  # For equal axis values
    
    # Calculate new limits
    limits = {
        "x_min": mins[0] - ranges[0] * margin,
        "x_max": mins[0] - ranges[0] * margin + max_range,
        "y_min": mins[1] - ranges[1] * margin,
        "y_max": mins[1] - ranges[1] * margin + max_range,
        "z_min": mins[2] - ranges[2] * margin,
        "z_max": mins[2] - ranges[2] * margin + max_range,
        "floor_y": mins[1] - ranges[1] * margin  # Floor at smallest y value y
    }
    
    return limits


def plot_skeleton_frame(axis, joint_position, axis_limits, vertical_viewing_angle=30, horizontal_rotation=45, graph_title=None):
    # Set labels
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_zlabel("Y")  # Y is up 

    # Set title if provided
    if graph_title:
        axis.set_title(graph_title)

    # Draw and plot floor surface 
    floor_x = np.linspace(axis_limits["x_min"], axis_limits["x_max"], 10)
    floor_z = np.linspace(axis_limits["z_min"], axis_limits["z_max"], 10)
    floor_x, floor_z = np.meshgrid(floor_x, floor_z)
    floor_y_values = np.ones_like(floor_x) * axis_limits["floor_y"]

    axis.plot_surface(floor_x, floor_z, floor_y_values, alpha=0.2, color="gray")

    # Plot joints with dots
    axis.scatter(
        joint_position[:, 0],
        joint_position[:, 2],
        joint_position[:, 1],
        c="black",
        marker="o",
        s=40,
        depthshade=True,
    )

    # Draw connections with appropriate colors
    for connection_group, color_name in [
        (torso_links, "torso"),
        (left_leg_links, "left_leg"),
        (right_leg_links, "right_leg"),
        (left_arm_links, "left_arm"),
        (right_arm_links, "right_arm")
    ]:
        for start, end in connection_group:
            axis.plot(
                [joint_position[start, 0], joint_position[end, 0]],
                [joint_position[start, 2], joint_position[end, 2]],
                [joint_position[start, 1], joint_position[end, 1]],
                color=COLORS[color_name],
                linewidth=3,
            )

    # Set axis limits
    axis.set_xlim(axis_limits["x_min"], axis_limits["x_max"])
    axis.set_ylim(axis_limits["z_min"], axis_limits["z_max"])
    axis.set_zlim(axis_limits["y_min"], axis_limits["y_max"])

    # Set view angle
    axis.view_init(elev=vertical_viewing_angle, azim=horizontal_rotation)


def create_gif(image_folder, output_path, duration=100):
    images = [
        os.path.join(image_folder, f)
        for f in sorted(os.listdir(image_folder))
        if f.endswith(".png")
    ]
    if not images:
        print("No PNG images found in the directory.")
        return
    frames = [imageio_v2.imread(image) for image in images]
    imageio_v2.mimsave(output_path, frames, duration=duration)  # type: ignore
    print(f"GIF saved as {output_path}")
