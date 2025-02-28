import torch
import numpy as np
from pathlib import Path
import argparse
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

def save_sequence_as_images(joints_data, output_dir="pose_frames", frame_interval=10, view_angles=None):
    """Save sequence as individual images instead of animation"""
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
    
    # Define default view angles if not provided
    if view_angles is None:
        view_angles = [(30, 45), (30, 135), (30, 225), (30, 315)]
    
    # Define connections for visualization (SMPL skeleton)
    connections = [
        # Torso
        (0, 1), (1, 4), (4, 7), (7, 10),  # Spine
        # Left leg
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        # Right leg
        (0, 2), (2, 5), (5, 8), (8, 11), (11, 14),
        # Left arm
        (7, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        # Right arm
        (7, 14), (14, 17), (17, 19), (19, 21), (21, 23)
    ]
    
    # Find data boundaries for setting axis limits
    min_val = np.min(joints)
    max_val = np.max(joints)
    buffer = (max_val - min_val) * 0.1  # Add 10% buffer
    
    # Frame count to save
    num_frames = len(joints)
    print(f"Saving frames at interval {frame_interval} from {num_frames} total frames")
    
    # Loop through frames
    for i in range(0, num_frames, frame_interval):
        joint_pos = joints[i]
        
        # Save each view angle as a separate image
        for view_idx, (elev, azim) in enumerate(view_angles):
            # Create figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {i} - View {view_idx}')
            
            # Set axis limits
            ax.set_xlim([min_val - buffer, max_val + buffer])
            ax.set_ylim([min_val - buffer, max_val + buffer])
            ax.set_zlim([min_val - buffer, max_val + buffer])
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            
            # Plot joints
            ax.scatter(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2], c='b', marker='o', s=30)
            
            # Plot connections
            for start, end in connections:
                ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                        [joint_pos[start, 1], joint_pos[end, 1]], 
                        [joint_pos[start, 2], joint_pos[end, 2]], 'r-')
            
            # Save figure
            plt.tight_layout()
            output_file = output_dir / f"frame_{i:04d}_view_{view_idx}.png"
            plt.savefig(output_file)
            plt.close(fig)
        
        if i % 50 == 0:
            print(f"Saved frame {i}/{num_frames}")
    
    print(f"Saved {len(list(output_dir.glob('*.png')))} frames to {output_dir}")
    return output_dir

def save_multi_view_grid(joints_data, output_path="pose_grid.png", frame_indices=None, view_angles=None):
    """Save a grid of multiple frames and viewpoints"""
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
        view_angles = [(30, 0), (30, 90), (30, 180), (30, 270)]
    
    # Define connections for visualization (SMPL skeleton)
    connections = [
        # Torso
        (0, 1), (1, 4), (4, 7), (7, 10),  # Spine
        # Left leg
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        # Right leg
        (0, 2), (2, 5), (5, 8), (8, 11), (11, 14),
        # Left arm
        (7, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        # Right arm
        (7, 14), (14, 17), (17, 19), (19, 21), (21, 23)
    ]
    
    # Create grid figure
    n_rows = len(frame_indices)
    n_cols = len(view_angles)
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    
    # Find data boundaries for setting axis limits
    min_val = np.min(joints)
    max_val = np.max(joints)
    buffer = (max_val - min_val) * 0.1  # Add 10% buffer
    
    # Loop through frames and views
    for i, frame_idx in enumerate(frame_indices):
        joint_pos = joints[frame_idx]
        
        for j, (elev, azim) in enumerate(view_angles):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i*n_cols + j + 1, projection='3d')
            
            # Set labels
            if i == n_rows-1:
                ax.set_xlabel('X')
            if j == 0:
                ax.set_ylabel('Y')
            
            # Set title
            ax.set_title(f'Frame {frame_idx} - View ({elev}°, {azim}°)')
            
            # Set axis limits
            ax.set_xlim([min_val - buffer, max_val + buffer])
            ax.set_ylim([min_val - buffer, max_val + buffer])
            ax.set_zlim([min_val - buffer, max_val + buffer])
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            
            # Plot joints
            ax.scatter(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2], c='b', marker='o', s=20)
            
            # Plot connections
            for start, end in connections:
                ax.plot([joint_pos[start, 0], joint_pos[end, 0]], 
                        [joint_pos[start, 1], joint_pos[end, 1]], 
                        [joint_pos[start, 2], joint_pos[end, 2]], 'r-')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved multi-view grid to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Visualize IMUPoser predictions")
    parser.add_argument("--predictions", type=str, default="pose_predictions.pt", help="Path to pose predictions file")
    parser.add_argument("--smpl_model", type=str, default=None, help="Path to SMPL model file")
    parser.add_argument("--output_dir", type=str, default="pose_frames", help="Directory to save pose frames")
    parser.add_argument("--grid", type=str, default="pose_grid.png", help="Path to save multi-view grid")
    parser.add_argument("--frame_interval", type=int, default=10, help="Interval between saved frames")
    args = parser.parse_args()
    
    # Setup environment
    root_dir = setup_environment()
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    
    # Convert poses to joint positions
    joints = convert_poses_to_joints(predictions, args.smpl_model)
    
    # Save sequence as images
    save_sequence_as_images(joints, args.output_dir, args.frame_interval)
    
    # Create multi-view grid of selected frames
    frame_indices = [0, 100, 200, 300, 400, 500] if joints.shape[1] >= 500 else None
    save_multi_view_grid(joints, args.grid, frame_indices)
    
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main()