import torch
from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def setup_environment():
    """Set up the Python environment by adding necessary paths"""
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))
    sys.path.append(str(root_dir / "src"))
    
    # Import necessary modules
    from imuposer.smpl.parametricModel import ParametricModel
    from imuposer.math.angular import r6d_to_rotation_matrix
    
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

def create_animation(joints_data, output_path="animation.mp4", fps=30, interval=33.33, downsample=1):
    """Create animation of the skeleton movement"""
    # Take the first sequence if multiple sequences
    if joints_data.shape[0] > 1:
        joints = joints_data[0]
    else:
        joints = joints_data.squeeze(0)
    
    # Downsample if needed
    if downsample > 1:
        joints = joints[::downsample]
    
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
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Joint scatter plot
    scatter = ax.scatter([], [], [], c='b', marker='o', s=20)
    
    # Lines for connections
    lines = [ax.plot([], [], [], 'r-')[0] for _ in connections]
    
    # Find data boundaries for setting axis limits
    min_val = joints.min().item()
    max_val = joints.max().item()
    buffer = (max_val - min_val) * 0.1  # Add 10% buffer
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('IMUPoser: 3D Joint Positions')
    
    def init():
        ax.set_xlim([min_val - buffer, max_val + buffer])
        ax.set_ylim([min_val - buffer, max_val + buffer])
        ax.set_zlim([min_val - buffer, max_val + buffer])
        
        scatter._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        
        return [scatter] + lines
    
    def update(frame):
        # Update joint positions
        joint_pos = joints[frame]
        x, y, z = joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2]
        scatter._offsets3d = (x, y, z)
        
        # Update connections
        for i, (start, end) in enumerate(connections):
            xs = [joint_pos[start, 0], joint_pos[end, 0]]
            ys = [joint_pos[start, 1], joint_pos[end, 1]]
            zs = [joint_pos[start, 2], joint_pos[end, 2]]
            lines[i].set_data(xs, ys)
            lines[i].set_3d_properties(zs)
        
        # Adjust view angle to create rotation effect
        ax.view_init(elev=30, azim=frame % 360)
        
        return [scatter] + lines
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=range(len(joints)),
        init_func=init, blit=False, interval=interval
    )
    
    print(f"Saving animation to {output_path}...")
    ani.save(output_path, fps=fps, dpi=100, extra_args=['-vcodec', 'libx264'])
    print(f"Animation saved successfully to {output_path}")
    
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Visualize IMUPoser predictions")
    parser.add_argument("--predictions", type=str, default="pose_predictions.pt", help="Path to pose predictions file")
    parser.add_argument("--smpl_model", type=str, default=None, help="Path to SMPL model file")
    parser.add_argument("--output", type=str, default="pose_animation.mp4", help="Path to save animation")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor for animation")
    args = parser.parse_args()
    
    # Setup environment
    root_dir = setup_environment()
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    
    # Convert poses to joint positions
    joints = convert_poses_to_joints(predictions, args.smpl_model)
    
    # Create and save animation
    create_animation(joints, args.output, args.fps, 1000/args.fps, args.downsample)
    
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main()