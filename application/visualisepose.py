# Modified version of the visualization code with dots but no labels

def save_multi_view_grid_with_dots(joints_data, output_path="pose_grid_dots.png", frame_indices=None, view_angles=None):
    """Save a grid of multiple frames and viewpoints with floor and joint dots but no labels"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Take the first sequence if multiple sequences
    if joints_data.shape[0] > 1:
        joints = joints_data[0]
    else:
        joints = joints_data.squeeze(0)
    
    # Convert to numpy for matplotlib
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy()
    
    # Define default frame indices if not provided (evenly spaced)
    if frame_indices is None:
        total_frames = len(joints)
        frame_indices = [int(i * total_frames / 6) for i in range(6)]
    
    # Define default view angles if not provided
    if view_angles is None:
        # Using conventional view angles
        view_angles = [
            (30, 0),    # Front view
            (30, 90),   # Side view (left)
        ]
    
    # Corrected connections for visualization (SMPL skeleton)
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
    
    # Colors for different body parts
    colors = {
        'left_leg': 'red',
        'torso': 'green',
        'right_leg': 'blue',
        'left_arm': 'orange',
        'right_arm': 'purple'
    }
    
    # Create grid figure
    n_rows = len(frame_indices)
    n_cols = len(view_angles)
    fig = plt.figure(figsize=(n_cols * 8, n_rows * 8))
    
    # Find data boundaries for setting axis limits
    # Calculate global min/max across all frames to keep consistent scaling
    all_joints = joints.reshape(-1, 3)
    
    # Find min/max for each axis
    x_min, y_min, z_min = np.min(all_joints, axis=0)
    x_max, y_max, z_max = np.max(all_joints, axis=0)
    
    # Add margin
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
    
    # Find the minimum y value for the floor
    floor_y = y_min
    
    # Loop through frames and views
    for i, frame_idx in enumerate(frame_indices):
        joint_pos = joints[frame_idx]
        
        for j, (elev, azim) in enumerate(view_angles):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i*n_cols + j + 1, projection='3d')
            
            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')  # Y is up in this coordinate system
            
            # Set title
            ax.set_title(f'Frame {frame_idx} - View ({elev}°, {azim}°)')
            
            # Draw the floor (a grid at the minimum y value)
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
                
            # Set axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            ax.set_zlim(y_min, y_max)
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            
            # Try to make the plot more square-like
            x_size = x_max - x_min
            y_size = y_max - y_min
            z_size = z_max - z_min
            max_size = max(x_size, y_size, z_size)
            ax.set_xlim(x_min, x_min + max_size)
            ax.set_ylim(z_min, z_min + max_size)
            ax.set_zlim(y_min, y_min + max_size)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved multi-view grid with dots to {output_path}")
    return output_path

def save_animated_sequence_with_dots(joints_data, output_dir="pose_frames_dots", frame_interval=5):
    """Save sequence frames for animation with floor and joint dots but no labels"""
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
    
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin  # Extra margin at bottom for floor
    y_max += y_range * margin
    z_min -= z_range * margin
    z_max += z_range * margin
    
    # Find the minimum y value for the floor
    floor_y = y_min
    
    # Make the axes have equal scales
    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min
    max_size = max(x_size, y_size, z_size)
    x_max = x_min + max_size
    z_max = z_min + max_size
    y_max = y_min + max_size
    
    # Set fixed view angle for animation
    elev, azim = 30, 45  # Default angle
    
    # Save frames
    for i in range(0, len(joints), frame_interval):
        joint_pos = joints[i]
        
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
        
        # Set axis limits
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
            print(f"Saved frame {i}/{len(joints)}")
    
    print(f"Saved frames with dots to {output_dir}")
    print(f"To create video, use command:")
    print(f"ffmpeg -framerate 30 -pattern_type glob -i '{output_dir}/frame_*.png' -c:v libx264 -pix_fmt yuv420p dots_animation.mp4")
    
    return output_dir

def main_with_dots():
    """Main function to visualize skeletal data with dots but no labels"""
    import argparse
    from pathlib import Path
    import sys
    
    parser = argparse.ArgumentParser(description="Visualize IMUPoser predictions with dots but no labels")
    parser.add_argument("--predictions", type=str, default="pose_predictions.pt", help="Path to pose predictions file")
    parser.add_argument("--smpl_model", type=str, default=None, help="Path to SMPL model file")
    parser.add_argument("--grid", type=str, default="pose_grid_dots.png", help="Path to save multi-view grid")
    parser.add_argument("--frames", type=str, default="pose_frames_dots", help="Directory to save individual frames")
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
        import torch
        
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
        import torch
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
    
    # Save multi-view grid with dots
    frame_indices = [0, 100, 200, 300, 400, 500] if joints.shape[1] >= 500 else None
    save_multi_view_grid_with_dots(joints, args.grid, frame_indices)
    
    # Save frames for animation with dots
    save_animated_sequence_with_dots(joints, args.frames, args.frame_interval)
    
    print("Visualization with dots completed successfully!")

if __name__ == "__main__":
    main_with_dots()