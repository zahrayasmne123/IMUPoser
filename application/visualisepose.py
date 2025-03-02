import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import imageio.v2 as imageio_v2

from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix

base_dir = "/dcs/22/u2254377/cs310/IMUPoser"

COLORS = {
    "torso": "green",
    "left_leg": "red",
    "right_leg": "blue",
    "left_arm": "orange",
    "right_arm": "purple",
}

TORSO_CONNECTIONS = [
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
]  # Pelvis to spine1 to spine2 to spine3 to neck to head

LEFT_LEG_CONNECTIONS = [
    (0, 1),
    (1, 4),
    (4, 7),
    (7, 10),
]  # Pelvis to L_hip to L_knee to L_ankle to L_foot

RIGHT_LEG_CONNECTIONS = [
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
]  # Pelvis to R_hip to R_knee to R_ankle to R_foot

LEFT_ARM_CONNECTIONS = [
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
]  # Spine3 to L_collar to L_shoulder to L_elbow to L_wrist to L_hand

RIGHT_ARM_CONNECTIONS = [
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
]  # Spine3 to R_collar to R_shoulder to R_elbow to R_wrist to R_hand



def prepare_joints_data(joints_data):
    # Take the first sequence if multiple sequences
    if joints_data.shape[0] > 1:
        joints = joints_data[0]
    else:
        joints = joints_data.squeeze(0)

    # Convert to numpy for matplotlib
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy()
        
    return joints

# Calculate axis limits for consistent visualisation.
def calculate_axis_limits(joints):
    # Calculate global min/max across all frames to keep consistent scaling
    all_joints = joints.reshape(-1, 3)

    
    x_min, y_min, z_min = np.min(all_joints, axis=0)
    x_max, y_max, z_max = np.max(all_joints, axis=0)


    margin = 0.1  # 10% margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin  # Extra margin at bottom for floor
    y_max += y_range * margin
    z_min -= z_range * margin
    z_max += z_range * margin

    # Make the axes have equal scales
    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min
    max_size = max(x_size, y_size, z_size)
    
    return {
        "x_min": x_min,
        "x_max": x_min + max_size,
        "y_min": y_min,
        "y_max": y_min + max_size,
        "z_min": z_min,
        "z_max": z_min + max_size,
        "floor_y": y_min  # Floor is at the minimum y value
    }

 
def plot_skeleton_frame(ax, joint_pos, axis_limits, elev=30, azim=45, title=None):
   
    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")  # Y is up in this coordinate system

    # Set title if provided
    if title:
        ax.set_title(title)

    # Draw the floor (a grid at the minimum y value)
    floor_x = np.linspace(axis_limits["x_min"], axis_limits["x_max"], 10)
    floor_z = np.linspace(axis_limits["z_min"], axis_limits["z_max"], 10)
    floor_x, floor_z = np.meshgrid(floor_x, floor_z)
    floor_y_values = np.ones_like(floor_x) * axis_limits["floor_y"]

    # Plot floor surface
    ax.plot_surface(floor_x, floor_z, floor_y_values, alpha=0.2, color="gray")

    # Plot joints with dots
    ax.scatter(
        joint_pos[:, 0],
        joint_pos[:, 2],
        joint_pos[:, 1],
        c="black",
        marker="o",
        s=40,
        depthshade=True,
    )

    # Draw connections with appropriate colors
    for connection_group, color_name in [
        (TORSO_CONNECTIONS, "torso"),
        (LEFT_LEG_CONNECTIONS, "left_leg"),
        (RIGHT_LEG_CONNECTIONS, "right_leg"),
        (LEFT_ARM_CONNECTIONS, "left_arm"),
        (RIGHT_ARM_CONNECTIONS, "right_arm")
    ]:
        for start, end in connection_group:
            ax.plot(
                [joint_pos[start, 0], joint_pos[end, 0]],
                [joint_pos[start, 2], joint_pos[end, 2]],
                [joint_pos[start, 1], joint_pos[end, 1]],
                color=COLORS[color_name],
                linewidth=3,
            )

    # Set axis limits
    ax.set_xlim(axis_limits["x_min"], axis_limits["x_max"])
    ax.set_ylim(axis_limits["z_min"], axis_limits["z_max"])
    ax.set_zlim(axis_limits["y_min"], axis_limits["y_max"])

    # Set view angle
    ax.view_init(elev=elev, azim=azim)


def save_multi_view_grid(
    joints_data, output_path="pose_grid.png", frame_indices=None, view_angles=None
):
    # Prepare joint data
    joints = prepare_joints_data(joints_data)
    
    # Define default frame indices if not provided (evenly spaced)
    if frame_indices is None:
        total_frames = len(joints)
        frame_indices = [int(i * total_frames / 6) for i in range(6)]

    # Define default view angles if not provided
    if view_angles is None:
        # Using conventional view angles
        view_angles = [
            (30, 0),  # Front view
            (30, 90),  # Side view (left)
        ]

    # Calculate axis limits
    axis_limits = calculate_axis_limits(joints)

    # Create grid figure
    n_rows = len(frame_indices)
    n_cols = len(view_angles)
    fig = plt.figure(figsize=(n_cols * 8, n_rows * 8))

    # Loop through frames and views
    for i, frame_idx in enumerate(frame_indices):
        joint_pos = joints[frame_idx]

        for j, (elev, azim) in enumerate(view_angles):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            
            # Plot skeleton
            plot_skeleton_frame(
                ax, 
                joint_pos, 
                axis_limits, 
                elev=elev, 
                azim=azim, 
                title=f"Frame {frame_idx} - View ({elev}°, {azim}°)"
            )

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved multi-view grid to {output_path}")
    return output_path


def save_animated_sequence_with_dots(
    joints_data, output_dir="pose_frames_dots", frame_interval=5
):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Prepare joint data
    joints = prepare_joints_data(joints_data)
    
    # Calculate axis limits
    axis_limits = calculate_axis_limits(joints)

    # Set fixed view angle for animation
    elev, azim = 30, 45  # Default angle

    # Save frames
    for i in range(0, len(joints), frame_interval):
        joint_pos = joints[i]

        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot skeleton
        plot_skeleton_frame(
            ax, 
            joint_pos, 
            axis_limits, 
            elev=elev, 
            azim=azim, 
            title=f"Frame {i}"
        )

        # Save figure
        plt.tight_layout()
        output_file = output_dir / f"frame_{i:04d}.png"
        plt.savefig(output_file, dpi=150)
        plt.close(fig)

        if i % 50 == 0:
            print(f"Saved frame {i}/{len(joints)}")

    print(f"Saved frames with dots to {output_dir}")
    return output_dir



# Convert poses to joint positions
def convert_poses_to_joints(
    predictions,
    smpl_model_path= os.path.join(base_dir, "/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
):
    """Convert pose predictions to 3D joint positions using SMPL"""

    # If SMPL model path not provided, use default
    if smpl_model_path is None:
        root_dir = Path(__file__).resolve().parent.parent
        smpl_model_path = root_dir / "src/imuposer/smpl/model.pkl"
        if not smpl_model_path.exists():
            smpl_model_path = (
                root_dir / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
            )

    print(f"Using SMPL model from: {smpl_model_path}")

    # Initialize SMPL model
    body_model = ParametricModel(str(smpl_model_path), device=torch.device("cpu"))

    # Get batch size and sequence length
    batch_size, seq_len, n_params = predictions.shape

    # Convert 6D rotation to rotation matrices (if needed)
    if n_params == 144:  # 24 joints * 6 params (6D representation)
        print("Converting 6D rotations to rotation matrices...")
        # Reshape to [batch_size*seq_len, num_joints*6]
        poses_reshape = predictions.reshape(-1, n_params)
        # Convert to rotation matrices [batch_size*seq_len, 216] (24 joints * 9 params)
        poses_matrices = r6d_to_rotation_matrix(poses_reshape).reshape(-1, 216)
    else:
        # Assume already in rotation matrix format
        poses_matrices = predictions.reshape(-1, 216)  # [batch_size*seq_len, 216]

    print(
        f"Computing joint positions for {batch_size} sequences with {seq_len} frames each..."
    )

    # Get joint positions from SMPL
    # This returns global poses and joint positions
    result = body_model.forward_kinematics(pose=poses_matrices)
    if len(result) == 2:
        _, joints = result
    else:
        _, joints, _ = result

    # Reshape back to [batch_size, seq_len, num_joints, 3]
    joints = joints.reshape(batch_size, seq_len, -1, 3)

    print(f"Joint positions computed with shape: {joints.shape}")
    return joints


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

def load_predictions(predictions_path):
    """Load the pose predictions from file"""
    import torch

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}")

    print(f"Loading predictions from: {predictions_path}")
    predictions = torch.load(predictions_path, map_location="cpu")

    print(f"Predictions shape: {predictions.shape}")
    return predictions

def visuals_pipeline():
    """Main function to visualize skeletal data with dots but no labels"""
    # Then use os.path.join for all paths
    predictions = load_predictions(os.path.join(base_dir, "rawdata/processed/predictions.pt"))
    joints = convert_poses_to_joints(
        predictions, os.path.join(base_dir, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    )
    
    # Save multi-view grid with dots
    frame_indices = [0, 100, 200, 300, 400, 500] if joints.shape[1] >= 500 else None
    save_multi_view_grid(joints, os.path.join(base_dir, "output/pose_grid_dots.png"), frame_indices)
    frames_dir = save_animated_sequence_with_dots(joints, os.path.join(base_dir, "output/pose_frames_dots"), 5)
    create_gif(frames_dir, os.path.join(base_dir, "output/output_frames.gif"))

    print("visualisation with dots completed successfully!")
