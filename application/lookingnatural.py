# Modified animation function with simulated global movement
def save_animated_sequence_with_natural_movement(joints_data, output_dir="pose_frames_natural", frame_interval=5):
    """Save sequence frames for animation with simulated global movement"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
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
    
    # Define colored connections for visualization
    colors = {
        'torso': 'green',
        'left_leg': 'red',
        'right_leg': 'blue',
        'left_arm': 'orange',
        'right_arm': 'purple'
    }
    
    # Corrected connections by body part
    # Spine/torso connections
    torso_connections = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]  # Pelvis to spine1 to spine2 to spine3 to neck to head
    
    # Left leg connections
    left_leg_connections = [(0, 1), (1, 4), (4, 7), (7, 10)]  # Pelvis to L_hip to L_knee to L_ankle to L_foot
    
    # Right leg connections
    right_leg_connections = [(0, 2), (2, 5), (5, 8), (8, 11)]  # Pelvis to R_hip to R_knee to R_ankle to R_foot
    
    # Left arm connections
    left_arm_connections = [(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)]  # Spine3 to L_collar to L_shoulder to L_elbow to L_wrist to L_hand
    
    # Right arm connections
    right_arm_connections = [(9, 14), (14, 17), (17, 19), (19, 21), (21, 23)]  # Spine3 to R_collar to R_shoulder to R_elbow to R_wrist to R_hand
    
    # Find data boundaries
    all_joints = joints.reshape(-1, 3)
    
    # Find min/max for each axis
    x_min, y_min, z_min = np.min(all_joints, axis=0)
    x_max, y_max, z_max = np.max(all_joints, axis=0)
    
    # Add margin
    margin = 0.1  # 10% margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    total_frames = len(joints)
    
    # Set fixed view angle for animation
    elev, azim = 30, 45  # Default angle
    
    # Determine movement direction based on foot positions
    # This is a simple heuristic to guess the forward direction
    try:
        # Look at a few frames to determine average direction
        sample_frames = [int(total_frames * 0.25), int(total_frames * 0.5), int(total_frames * 0.75)]
        foot_positions = []
        for i in sample_frames:
            if i < total_frames:
                # Average of left and right foot positions
                foot_pos = (joints[i, 10, :] + joints[i, 11, :]) / 2
                foot_positions.append(foot_pos)
        
        if len(foot_positions) > 1:
            # Calculate average direction vector
            directions = []
            for i in range(1, len(foot_positions)):
                direction = foot_positions[i] - foot_positions[i-1]
                if np.linalg.norm(direction) > 0.01:  # Only consider significant movements
                    directions.append(direction)
            
            if directions:
                avg_direction = np.mean(directions, axis=0)
                avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-10)  # Normalize
            else:
                # Default forward direction if we couldn't determine
                avg_direction = np.array([0, 0, 1])
        else:
            # Default forward direction if we don't have enough frames
            avg_direction = np.array([0, 0, 1])
    except:
        # Fallback if anything goes wrong
        avg_direction = np.array([0, 0, 1])
    
    # Normalize the direction vector
    forward_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-10)
    
    # Calculate global translation for each frame to simulate movement
    # We'll use the root joint (pelvis) as reference
    root_positions = joints[:, 0, :]  # Get all pelvis positions
    
    # Calculate foot contact pattern to make movement more natural
    # We'll use a simplified approach by checking the height of feet
    left_foot_heights = joints[:, 10, 1]  # Y is up
    right_foot_heights = joints[:, 11, 1]
    
    # Find average foot height when standing
    avg_foot_height = np.percentile(np.concatenate([left_foot_heights, right_foot_heights]), 10)
    foot_threshold = avg_foot_height + 0.05  # Threshold for foot contact
    
    # Determine when feet are in contact with the ground
    left_contact = left_foot_heights < foot_threshold
    right_contact = right_foot_heights < foot_threshold
    any_contact = np.logical_or(left_contact, right_contact)
    
    # Estimate natural translation based on foot contact and root motion
    translations = np.zeros((total_frames, 3))
    step_size = 0.01  # Base step size
    
    for i in range(1, total_frames):
        if any_contact[i]:
            # When foot is in contact, move in forward direction
            translations[i] = translations[i-1] + forward_direction * step_size
        else:
            # Otherwise keep same position as previous frame
            translations[i] = translations[i-1]
    
    # Set up axis limits that will work for all frames
    # We'll use a moving window that follows the character
    window_size = max(x_range, z_range) * 1.5
    
    # Save frames
    for i in range(0, total_frames, frame_interval):
        joint_pos = joints[i].copy()  # Copy to avoid modifying original data
        
        # Apply the estimated global translation to make movement look natural
        joint_pos += translations[i]
        
        # Calculate window center based on current position
        center_x = joint_pos[0, 0]  # Root joint x position
        center_z = joint_pos[0, 2]  # Root joint z position
        
        # Set up axis limits for this frame
        x_min = center_x - window_size/2
        x_max = center_x + window_size/2
        z_min = center_z - window_size/2
        z_max = center_z + window_size/2
        
        # Y limits remain constant (height)
        y_min = np.min(all_joints[:, 1]) - 0.1 * y_range
        y_max = np.max(all_joints[:, 1]) + 0.1 * y_range
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')  # Y is up
        
        # Set title
        ax.set_title(f'Frame {i}')
        
        # Draw the floor (a grid at the minimum y value)
        floor_y = y_min
        floor_x = np.linspace(x_min, x_max, 10)
        floor_z = np.linspace(z_min, z_max, 10)
        floor_x, floor_z = np.meshgrid(floor_x, floor_z)
        floor_y_values = np.ones_like(floor_x) * floor_y
        
        # Plot floor surface
        ax.plot_surface(floor_x, floor_z, floor_y_values, alpha=0.2, color='gray')
        
        # Plot joints with dots
        ax.scatter(joint_pos[:, 0], joint_pos[:, 2], joint_pos[:, 1], 
                  c='black', marker='o', s=40, depthshade=True)
        
        # Draw connections with appropriate colors
        for start, end in torso_connections:
            ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                    [joint_pos[start, 2], joint_pos[end, 2]], 
                    [joint_pos[start, 1], joint_pos[end, 1]], 
                    color=colors['torso'], linewidth=3)
        
        for start, end in left_leg_connections:
            ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                    [joint_pos[start, 2], joint_pos[end, 2]], 
                    [joint_pos[start, 1], joint_pos[end, 1]], 
                    color=colors['left_leg'], linewidth=3)
        
        for start, end in right_leg_connections:
            ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                    [joint_pos[start, 2], joint_pos[end, 2]], 
                    [joint_pos[start, 1], joint_pos[end, 1]], 
                    color=colors['right_leg'], linewidth=3)
        
        for start, end in left_arm_connections:
            ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                    [joint_pos[start, 2], joint_pos[end, 2]], 
                    [joint_pos[start, 1], joint_pos[end, 1]], 
                    color=colors['left_arm'], linewidth=3)
        
        for start, end in right_arm_connections:
            ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                    [joint_pos[start, 2], joint_pos[end, 2]], 
                    [joint_pos[start, 1], joint_pos[end, 1]], 
                    color=colors['right_arm'], linewidth=3)
        
        # Set axis limits - follow the character
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_zlim(y_min, y_max)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Save figure
        plt.tight_layout()
        output_file = output_dir / f"frame_{i:04d}.png"
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        
        if i % 50 == 0:
            print(f"Saved frame {i}/{total_frames}")
    
    print(f"Saved frames with natural movement to {output_dir}")
    print(f"To create video, use command:")
    print(f"ffmpeg -framerate 30 -pattern_type glob -i '{output_dir}/frame_*.png' -c:v libx264 -pix_fmt yuv420p natural_movement.mp4")
    
    return output_dir


# Function to run the natural movement visualization
def main_with_natural_movement():
    """Main function to visualize skeletal data with natural movement"""
    import argparse
    from pathlib import Path
    import sys
    import torch
    
    parser = argparse.ArgumentParser(description="Visualize IMUPoser predictions with natural movement")
    parser.add_argument("--predictions", type=str, default="pose_predictions.pt", help="Path to pose predictions file")
    parser.add_argument("--smpl_model", type=str, default=None, help="Path to SMPL model file")
    parser.add_argument("--frames", type=str, default="pose_frames_natural", help="Directory to save individual frames")
    parser.add_argument("--frame_interval", type=int, default=5, help="Interval between saved frames")
    args = parser.parse_args()
    
    # Setup environment
    def setup_environment():
        """Set up the Python environment by adding necessary paths"""
        root_dir = Path(__file__).resolve().parent.parent
        sys.path.append(str(root_dir))
        sys.path.append(str(root_dir / "src"))
        return root_dir
    
    root_dir = setup_environment()
    
    # Load predictions
    def load_predictions(predictions_path):
        """Load the pose predictions from file"""
        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
        
        print(f"Loading predictions from: {predictions_path}")
        predictions = torch.load(predictions_path, map_location='cpu')
        
        print(f"Predictions shape: {predictions.shape}")
        return predictions
    
    predictions = load_predictions(args.predictions)
    
    # Convert poses to joint positions
    def convert_poses_to_joints(predictions, smpl_model_path=None):
        """Convert pose predictions to 3D joint positions using SMPL"""
        from pathlib import Path
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
    
    joints = convert_poses_to_joints(predictions, args.smpl_model)
    
    # Save frames for animation with natural movement
    save_animated_sequence_with_natural_movement(joints, args.frames, args.frame_interval)
    
    print("Natural movement visualization completed successfully!")

if __name__ == "__main__":
    main_with_natural_movement()