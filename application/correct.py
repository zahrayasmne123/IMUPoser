import torch
import numpy as np
from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def setup_environment():
    """Set up the Python environment by adding necessary paths"""
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))
    sys.path.append(str(root_dir / "src"))
    return root_dir

def load_predictions(predictions_path):
    """Load the pose predictions from file"""
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
    
    print(f"Loading predictions from: {predictions_path}")
    predictions = torch.load(predictions_path, map_location='cpu')
    
    print(f"Predictions shape: {predictions.shape}")
    return predictions

def convert_poses_to_joints(predictions, smpl_model_path=None):
    """Convert pose predictions to 3D joint positions using SMPL"""
    from imuposer.smpl.parametricModel import ParametricModel
    from imuposer.math.angular import r6d_to_rotation_matrix
    
    # If SMPL model path not provided, use default
    if smpl_model_path is None:
        root_dir = Path(__file__).resolve().parent.parent
        smpl_model_path = root_dir / "src/imuposer/smpl/model.pkl"
        if not smpl_model_path.exists():
            smpl_model_path = root_dir / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    
    print(f"Using SMPL model from: {smpl_model_path}")
    
    # Initialize SMPL model
    body_model = ParametricModel(str(smpl_model_path), device=torch.device('cpu'))
    
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
    
    print(f"Computing joint positions for {batch_size} sequences with {seq_len} frames each...")
    
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

def print_joint_indices(joint_pos):
    """Print the joint indices for debugging"""
    print("Joint indices:")
    for i in range(joint_pos.shape[0]):
        print(f"Joint {i}: {joint_pos[i]}")

def save_single_frame(joint_pos, output_path, view_angle=(30, 45), print_indices=False):
    """Save a single frame visualization with multiple views and correct connections"""
    
    # Print joint indices if requested
    if print_indices:
        print_joint_indices(joint_pos)
    
    # Colors for different body parts
    colors = {
        'torso': 'red',
        'left_leg': 'green',
        'right_leg': 'blue',
        'left_arm': 'orange',
        'right_arm': 'purple'
    }
    
    # SMPL joint indices reference:
    # 0: pelvis/root
    # 1, 2: left/right hip
    # 4, 5: left/right knee
    # 7, 8: left/right ankle
    # 10: spine1
    # 11: spine2
    # 14: neck
    # 15: head
    # 16, 17: left/right shoulder
    # 18, 19: left/right elbow
    # 20, 21: left/right wrist
    
    # Corrected connections by body part
    torso_connections = [(0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]  # Pelvis to head
    left_leg_connections = [(1, 4), (4, 7), (7, 10)]  # Left hip to ankle
    right_leg_connections = [(2, 5), (5, 8), (8, 11)]  # Right hip to ankle
    left_arm_connections = [(9, 13), (13, 16), (16, 18), (18, 20)]  # Spine to left wrist
    right_arm_connections = [(9, 14), (14, 17), (17, 19), (19, 21)]  # Spine to right wrist
    
    # Create 2x2 grid with four different views
    fig = plt.figure(figsize=(16, 16))
    
    # Define four different views
    views = [
        (30, 0),    # Front view
        (30, 90),   # Side view (right)
        (30, 180),  # Back view
        (30, 270)   # Side view (left)
    ]
    
    # Find the minimum y value for the floor
    floor_y = np.min(joint_pos[:, 1]) - 0.1
    
    # Calculate axis limits
    margin = 0.2
    x_min, y_min, z_min = np.min(joint_pos, axis=0) - margin
    x_max, y_max, z_max = np.max(joint_pos, axis=0) + margin
    
    # Make the plot square
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    
    x_mid = (x_max + x_min) / 2
    z_mid = (z_max + z_min) / 2
    
    x_min, x_max = x_mid - max_range/2, x_mid + max_range/2
    z_min, z_max = z_mid - max_range/2, z_mid + max_range/2
    
    # Keep y as is to show the floor
    
    # Plot all four views
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Set title based on view
        view_name = {
            0: "Front View",
            90: "Right Side View",
            180: "Back View",
            270: "Left Side View"
        }.get(azim, f"View ({elev}°, {azim}°)")
        
        ax.set_title(view_name, fontsize=16)
        
        # Set labels
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Z', fontsize=12)
        ax.set_zlabel('Y', fontsize=12)  # Y is up
        
        # Draw the floor
        floor_x = np.linspace(x_min, x_max, 10)
        floor_z = np.linspace(z_min, z_max, 10)
        floor_x, floor_z = np.meshgrid(floor_x, floor_z)
        floor_y_values = np.ones_like(floor_x) * floor_y
        
        # Plot floor surface
        ax.plot_surface(floor_x, floor_z, floor_y_values, alpha=0.1, color='gray')
        
        # Plot joints with different colors for better visibility
        ax.scatter(joint_pos[:, 0], joint_pos[:, 2], joint_pos[:, 1], 
                  c='b', marker='o', s=50, depthshade=True)
        
        # Draw connections with appropriate colors
        # Draw torso connections in red
        for start, end in torso_connections:
            try:
                ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                        [joint_pos[start, 2], joint_pos[end, 2]], 
                        [joint_pos[start, 1], joint_pos[end, 1]], 
                        color=colors['torso'], linewidth=4)
            except IndexError:
                print(f"Error drawing connection between joints {start} and {end}")
        
        # Draw left leg connections in green
        for start, end in left_leg_connections:
            try:
                ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                        [joint_pos[start, 2], joint_pos[end, 2]], 
                        [joint_pos[start, 1], joint_pos[end, 1]], 
                        color=colors['left_leg'], linewidth=4)
            except IndexError:
                print(f"Error drawing connection between joints {start} and {end}")
        
        # Draw right leg connections in blue
        for start, end in right_leg_connections:
            try:
                ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                        [joint_pos[start, 2], joint_pos[end, 2]], 
                        [joint_pos[start, 1], joint_pos[end, 1]], 
                        color=colors['right_leg'], linewidth=4)
            except IndexError:
                print(f"Error drawing connection between joints {start} and {end}")
        
        # Draw left arm connections in orange
        for start, end in left_arm_connections:
            try:
                ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                        [joint_pos[start, 2], joint_pos[end, 2]], 
                        [joint_pos[start, 1], joint_pos[end, 1]], 
                        color=colors['left_arm'], linewidth=4)
            except IndexError:
                print(f"Error drawing connection between joints {start} and {end}")
        
        # Draw right arm connections in purple
        for start, end in right_arm_connections:
            try:
                ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                        [joint_pos[start, 2], joint_pos[end, 2]], 
                        [joint_pos[start, 1], joint_pos[end, 1]], 
                        color=colors['right_arm'], linewidth=4)
            except IndexError:
                print(f"Error drawing connection between joints {start} and {end}")
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_zlim(y_min, y_max)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color=colors['left_leg'], lw=4, label='Left Leg'),
            Line2D([0], [0], color=colors['left_leg'], lw=4, label='Left Leg'),
            Line2D([0], [0], color=colors['right_leg'], lw=4, label='Right Leg'),
            Line2D([0], [0], color=colors['left_arm'], lw=4, label='Left Arm'),
            Line2D([0], [0], color=colors['right_arm'], lw=4, label='Right Arm')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Add a centered title for the figure
    plt.suptitle("Pose Visualization", fontsize=20)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    return output_path

def save_key_frames(joints_data, output_dir="pose_frames", num_frames=10, print_first=False):
    """Save specific frames spread throughout the sequence"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Take the first sequence if multiple sequences
    if joints_data.shape[0] > 1:
        joints = joints_data[0]
    else:
        joints = joints_data.squeeze(0)
    
    # Convert to numpy for matplotlib
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy()
    
    # Calculate frame indices to save (evenly spaced)
    total_frames = len(joints)
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    print(f"Saving {len(frame_indices)} frames from sequence with {total_frames} total frames")
    
    # Save each frame
    for i, frame_idx in enumerate(frame_indices):
        joint_pos = joints[frame_idx]
        output_file = output_dir / f"frame_{frame_idx:04d}.png"
        
        # Print joint indices for the first frame if requested
        print_indices = print_first and i == 0
        
        save_single_frame(joint_pos, output_file, print_indices=print_indices)
        print(f"Saved frame {i+1}/{len(frame_indices)}: {output_file}")
    
    print(f"Saved {len(frame_indices)} frames to {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Create individual pose frame visualizations with corrected connections")
    parser.add_argument("--predictions", type=str, default="pose_predictions.pt", help="Path to pose predictions file")
    parser.add_argument("--smpl_model", type=str, default=None, help="Path to SMPL model file")
    parser.add_argument("--output_dir", type=str, default="pose_frames", help="Directory to save frames")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to save")
    parser.add_argument("--print_joints", action="store_true", help="Print joint indices for the first frame")
    args = parser.parse_args()
    
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    
    # Convert poses to joint positions
    joints = convert_poses_to_joints(predictions, args.smpl_model)
    
    # Save individual frames
    save_key_frames(joints, args.output_dir, args.num_frames, print_first=args.print_joints)
    
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main()