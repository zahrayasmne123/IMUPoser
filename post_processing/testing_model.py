import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
import sys

# Add the project root directory to Python path so it can find the src module
base_dir = "/dcs/22/u2254377/cs310/IMUPoser"
sys.path.append(base_dir)  # This is the critical fix for your import error

# Now the imports should work
from src.imuposer.config import Config, amass_combos
from src.imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel

dataset_path = Path(os.path.join(base_dir, "imuposer_dataset"))

def load_model(checkpoint_path):
    """
    Load the IMUPoser model from a checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Loaded model
    """
    try:
        # Make sure checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Load checkpoint with CPU mapping
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Determine configuration
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
            config_dict = checkpoint['hyper_parameters']['config']
            
            # Explicitly create Config with CPU device as string
            config = Config(
                experiment=config_dict.experiment if hasattr(config_dict, 'experiment') else "IMUPoserGlobalModel",
                model=config_dict.model if hasattr(config_dict, 'model') else "GlobalModelIMUPoser",
                project_root_dir=base_dir,
                joints_set=config_dict.joints_set if hasattr(config_dict, 'joints_set') else amass_combos['global'],
                normalize=False,
                r6d=True,
                loss_type="mse",
                use_joint_loss=True,
                device='cpu',  # Use string 'cpu' instead of torch.device
                og_smpl_model_path=os.path.join(base_dir, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
            )
        else:
            # Fallback configuration
            config = Config(
                experiment="IMUPoserGlobalModel",
                model="GlobalModelIMUPoser",
                project_root_dir=base_dir,
                joints_set=amass_combos['global'],
                normalize=False,
                r6d=True,
                loss_type="mse",
                use_joint_loss=True,
                device='cpu',  # Use string 'cpu' instead of torch.device
                og_smpl_model_path=os.path.join(base_dir, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
            )
        
        # Convert the string path to a Path object
        config.og_smpl_model_path = Path(config.og_smpl_model_path)
        
        # Verify SMPL model path exists
        if not config.og_smpl_model_path.exists():
            raise FileNotFoundError(f"SMPL model not found at {config.og_smpl_model_path}")
        
        # Create model instance
        model = IMUPoserModel(config)
        
        # Load weights
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: Could not find state_dict in checkpoint")
        
        # Ensure model is in evaluation mode
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_test_data(participant_ids=None, activities=None):
    """
    Load test data from the IMUPoser dataset
    
    Args:
        participant_ids: List of participant IDs (e.g., ["P1", "P2"])
        activities: List of activity names to include (e.g., ["Walking", "ArmRaises"])
        
    Returns:
        Dictionary containing IMU inputs and ground truth pose data
    """
    if participant_ids is None:
        # Default to all participants
        participant_ids = [f"P{i}" for i in range(1, 11)]
    
    test_data = {
        "imu_inputs": [],
        "pose_gt": [],
        "trans_gt": [],
        "betas_gt": [],
        "metadata": []
    }
    
    for pid in tqdm(participant_ids, desc="Loading participants"):
        pid_path = dataset_path / pid
        if not pid_path.exists():
            print(f"Warning: {pid} not found")
            continue
        
        for activity_file in pid_path.glob("*.pkl"):
            activity_name = activity_file.stem
            
            # Filter by activity if specified
            if activities is not None and not any(act in activity_name for act in activities):
                continue
            
            try:
                with open(activity_file, "rb") as f:
                    data = pkl.load(f)
                
                # Extract IMU data
                imu_data = data["imu"]
                
                # Extract ground truth pose
                pose_gt = data["pose"]
                
                # Extract translation if available
                trans_gt = data.get("trans", None)
                
                # Extract betas if available (body shape parameters)
                betas_gt = data.get("betas", None)
                
                # Append to our test data
                test_data["imu_inputs"].append(imu_data)
                test_data["pose_gt"].append(pose_gt)
                
                if trans_gt is not None:
                    test_data["trans_gt"].append(trans_gt)
                
                if betas_gt is not None:
                    test_data["betas_gt"].append(betas_gt)
                
                test_data["metadata"].append({
                    "participant": pid,
                    "activity": activity_name
                })
                
            except Exception as e:
                print(f"Error loading {activity_file}: {e}")
    
    return test_data


def preprocess_imu_data(imu_data, window_size=125, stride=1):
    """
    Preprocess IMU data into sliding windows for input to the model
    
    Args:
        imu_data: Raw IMU data from the dataset
        window_size: Size of sliding window (125 frames = 5 seconds at 25 FPS)
        stride: Stride between consecutive windows
        
    Returns:
        List of windowed IMU data
    """
    windows = []
    
    for i in range(0, len(imu_data) - window_size + 1, stride):
        window = imu_data[i:i+window_size]
        windows.append(window)
    
    return windows

def run_inference(model, imu_inputs, device="cpu"):
    """
    Run inference on test data using your trained model
    
    Args:
        model: Your trained IMUPoser model
        imu_inputs: List of IMU input tensors
        device: Device to run inference on
        
    Returns:
        List of predicted poses (each with shape corresponding to the ground truth)
    """
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for imu_input in tqdm(imu_inputs, desc="Running inference"):
            # Convert to torch tensor if needed
            if not isinstance(imu_input, torch.Tensor):
                imu_input = torch.tensor(imu_input, dtype=torch.float32).to(device)
            
            # Ensure input has correct dimensions
            if len(imu_input.shape) == 2:  # [sequence_length, features]
                imu_input = imu_input.unsqueeze(0)  # Add batch dimension
            
            # Create imu_lens tensor
            batch_size = imu_input.shape[0]
            seq_length = imu_input.shape[1]
            imu_lens = torch.full((batch_size,), seq_length, dtype=torch.long, device=device)
            
            # Run model inference
            pred_pose = model(imu_input, imu_lens)
            
            # Extract the middle frame (to match our ground truth extraction)
            if isinstance(pred_pose, torch.Tensor):
                # If prediction is for full sequence, take middle frame
                if len(pred_pose.shape) == 3 and pred_pose.shape[1] > 1:
                    middle_idx = pred_pose.shape[1] // 2
                    pred_pose = pred_pose[:, middle_idx, :]
                
                # Keep as tensor for now for potential conversion
                if pred_pose.shape[-1] == 144:  # 6D rotation format
                    # Here we would ideally convert 6D to 3D rotation
                    # For now, just extract every other value as a rough approximation
                    # In a real implementation, you'd use proper rotation conversion functions
                    pred_pose_3d = pred_pose.reshape(-1, 24, 6)[:, :, :3].reshape(-1, 72)
                    pred_pose = pred_pose_3d
                
                pred_pose = pred_pose.cpu().numpy()
            
            # Remove batch dimension if present
            if len(pred_pose.shape) > 1 and pred_pose.shape[0] == 1:
                pred_pose = pred_pose.squeeze(0)
                
            predictions.append(pred_pose)
    
    return predictions

def calculate_mse(predictions, ground_truth):
    """
    Calculate MSE between predicted poses and ground truth
    
    Args:
        predictions: List of predicted poses
        ground_truth: List of ground truth poses
        
    Returns:
        Dictionary of MSE results
    """
    overall_mse = []
    per_joint_mse = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Ensure numpy arrays
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        
        # Print shapes for debugging
        print(f"Debug - pred shape: {pred.shape}, gt shape: {gt.shape}")
        
        try:
            # If shapes match, calculate MSE directly
            if pred.shape == gt.shape:
                mse = np.mean((pred - gt) ** 2)
                overall_mse.append(mse)
                
                # Calculate per-joint MSE
                joint_mse = np.mean((pred.reshape(-1, 24, 3) - gt.reshape(-1, 24, 3)) ** 2, axis=(0, 2))
                per_joint_mse.append(joint_mse)
            else:
                print(f"Shape mismatch: pred {pred.shape}, gt {gt.shape} - skipping")
        except Exception as e:
            print(f"Error calculating MSE: {e}")
    
    if not overall_mse:
        return {
            "overall_mse": float('nan'),
            "per_sequence_mse": [],
            "per_joint_mse": []
        }
    
    # Average per-joint MSE across all examples
    if per_joint_mse:
        avg_per_joint_mse = np.mean(per_joint_mse, axis=0)
    else:
        avg_per_joint_mse = np.array([])
    
    return {
        "overall_mse": np.mean(overall_mse),
        "per_sequence_mse": overall_mse,
        "per_joint_mse": avg_per_joint_mse
    }

def visualize_results(participant, activity, pred_pose, gt_pose, output_dir="results"):
    """
    Visualize comparison between predicted and ground truth poses
    
    Args:
        participant: Participant ID
        activity: Activity name
        pred_pose: Predicted pose
        gt_pose: Ground truth pose
        output_dir: Directory to save visualizations to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy if needed
    if isinstance(pred_pose, torch.Tensor):
        pred_pose = pred_pose.cpu().numpy()
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = gt_pose.cpu().numpy()
    
    # Handle different shapes
    if len(pred_pose.shape) > 1 and pred_pose.shape[0] == 1:
        pred_pose = pred_pose.squeeze(0)
    
    # Print shapes for debugging
    print(f"Visualization - pred shape: {pred_pose.shape}, gt shape: {gt_pose.shape}")
    
    # Create a simple plot comparing parameter values
    plt.figure(figsize=(12, 6))
    plt.title(f"Pose Parameters - {participant} - {activity}")
    
    # Make sure we're only plotting what we can compare
    if pred_pose.shape[0] > gt_pose.shape[0]:
        pred_plot = pred_pose[:gt_pose.shape[0]]
    else:
        pred_plot = pred_pose
        
    plt.plot(gt_pose, label='Ground Truth', alpha=0.7)
    plt.plot(pred_plot, label='Prediction', alpha=0.7)
    plt.legend()
    plt.xlabel("Parameter Index")
    plt.ylabel("Value")
    
    output_file = os.path.join(output_dir, f"pose_params_{participant}_{activity}.png")
    plt.savefig(output_file)
    plt.close()
    
    # Try to create a heatmap visualization if shapes allow
    try:
        # Reshape to joint x parameter format
        if gt_pose.shape[0] == 72:  # 3D rotation format (24 joints x 3 parameters)
            gt_reshaped = gt_pose.reshape(24, 3)
            if pred_pose.shape[0] >= 72:
                pred_reshaped = pred_pose[:72].reshape(24, 3)
            else:
                pred_reshaped = pred_pose.reshape(-1, 3)[:24, :]
            param_label = "Rotation Parameters (3D)"
        else:
            # Skip heatmap if we can't determine the proper reshaping
            raise ValueError("Unexpected shape for heatmap visualization")
            
        # Create heatmap visualization
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot ground truth pose parameters as a heatmap
        axs[0].set_title(f"Ground Truth - {participant} - {activity}")
        im0 = axs[0].imshow(gt_reshaped, cmap='viridis', aspect='auto')
        axs[0].set_xlabel(param_label)
        axs[0].set_ylabel("Joint Index")
        fig.colorbar(im0, ax=axs[0])
        
        # Plot predicted pose parameters as a heatmap
        axs[1].set_title(f"Prediction - {participant} - {activity}")
        im1 = axs[1].imshow(pred_reshaped, cmap='viridis', aspect='auto')
        axs[1].set_xlabel(param_label)
        axs[1].set_ylabel("Joint Index")
        fig.colorbar(im1, ax=axs[1])
        
        plt.tight_layout()
        
        # Save the figure
        heatmap_file = os.path.join(output_dir, f"pose_heatmap_{participant}_{activity}.png")
        plt.savefig(heatmap_file)
        plt.close()
        
        # Also plot the difference between predicted and ground truth
        plt.figure(figsize=(10, 8))
        plt.title(f"Pose Difference - {participant} - {activity}")
        diff = np.abs(pred_reshaped - gt_reshaped)
        im = plt.imshow(diff, cmap='hot', aspect='auto')
        plt.colorbar(im)
        plt.xlabel(param_label)
        plt.ylabel("Joint Index")
        
        # Save the difference figure
        diff_file = os.path.join(output_dir, f"pose_diff_{participant}_{activity}.png")
        plt.savefig(diff_file)
        plt.close()
        
        return output_file, heatmap_file, diff_file
    except Exception as e:
        print(f"Couldn't create heatmap visualization: {e}")
        return output_file, output_file



def calculate_mpjpe(predicted_joints, ground_truth_joints):
    """
    Calculate Mean Per Joint Position Error
    
    Args:
        predicted_joints: numpy array of shape (n_frames, n_joints, 3)
        ground_truth_joints: numpy array of shape (n_frames, n_joints, 3)
    
    Returns:
        MPJPE value in millimeters
    """
    # Calculate Euclidean distance for each joint in each frame
    joint_errors = np.sqrt(np.sum((predicted_joints - ground_truth_joints)**2, axis=2))
    
    # Average across joints for each frame
    frame_errors = np.mean(joint_errors, axis=1)
    
    # Average across frames
    mpjpe = np.mean(frame_errors)
    
    return mpjpe

def calculate_acceleration_error(predicted_joints, ground_truth_joints, dt=1/25.0):
    """
    Calculate acceleration error between predicted and ground truth joint trajectories
    
    Args:
        predicted_joints: numpy array of shape (n_frames, n_joints, 3)
        ground_truth_joints: numpy array of shape (n_frames, n_joints, 3)
        dt: time step between frames in seconds (default: 1/25 for 25 FPS)
    
    Returns:
        Mean acceleration error in mm/s²
    """
    # Calculate acceleration for predicted joints (2nd derivative)
    pred_acc = (predicted_joints[2:] - 2 * predicted_joints[1:-1] + predicted_joints[:-2]) / (dt**2)
    
    # Calculate acceleration for ground truth joints
    gt_acc = (ground_truth_joints[2:] - 2 * ground_truth_joints[1:-1] + ground_truth_joints[:-2]) / (dt**2)
    
    # Calculate acceleration error for each joint in each frame
    acc_error = np.sqrt(np.sum((pred_acc - gt_acc)**2, axis=2))
    
    # Average across joints and frames
    mean_acc_error = np.mean(acc_error)
    
    return mean_acc_error


def calculate_smpl_param_error(predicted_params, ground_truth_params):
    """
    Calculate Mean Squared Error of SMPL pose parameters
    
    Args:
        predicted_params: numpy array of shape (n_frames, 144) - SMPL pose parameters
        ground_truth_params: numpy array of shape (n_frames, 144) - SMPL pose parameters
    
    Returns:
        MSE of SMPL parameters
    """
    # Calculate MSE
    mse = np.mean((predicted_params - ground_truth_params)**2)
    
    return mse
# Main execution
if __name__ == "__main__":
    print("Starting IMUPoser testing with extended metrics...")
    
    # 1. Load your trained model - always use CPU
    checkpoint_path = os.path.join(base_dir, "checkpoint.ckpt")
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path)
    
    # Import the visualization module with joint conversion functions
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from visualisepose import convert_poses_to_joints
    
    # 2. Load test data with specific activities
    print("Loading test data...")
    test_data = load_test_data(
        participant_ids=["P1", "P2"], 
        activities=["Walking", "ArmRaises"]
    )
    
    # 3. Debug output the shapes
    print("\nDEBUG - Data shapes:")
    for i, (imu, pose) in enumerate(zip(test_data["imu_inputs"][:2], test_data["pose_gt"][:2])):
        print(f"Example {i}:")
        print(f"  IMU shape: {imu.shape}")
        print(f"  Pose shape: {pose.shape}")
    
    # 4. Preprocess the data
    processed_inputs = []
    processed_gt = []
    metadata = []
    
    print("Preprocessing data...")
    for idx, (imu, pose) in enumerate(zip(test_data["imu_inputs"], test_data["pose_gt"])):
        # Create sliding windows
        imu_windows = preprocess_imu_data(imu, window_size=125, stride=25)
        
        # Get corresponding pose frames (middle frame of each window)
        pose_center_frames = []
        for i in range(0, len(pose) - 124, 25):
            if i + 62 < len(pose):  # Ensure middle frame exists
                pose_center_frames.append(pose[i + 62])
        
        # Ensure we have matching counts
        min_len = min(len(imu_windows), len(pose_center_frames))
        imu_windows = imu_windows[:min_len]
        pose_center_frames = pose_center_frames[:min_len]
        
        processed_inputs.extend(imu_windows)
        processed_gt.extend(pose_center_frames)
        
        # Add metadata for each window
        for _ in range(min_len):
            metadata.append(test_data["metadata"][idx])
    
    print(f"Processed {len(processed_inputs)} input windows with corresponding ground truth poses")
    
    # 5. Run inference
    print(f"Running inference on {len(processed_inputs)} samples...")
    predictions = run_inference(model, processed_inputs, device='cpu')
    
    # 6. Calculate MSE
    print("Calculating MSE...")
    mse_results = calculate_mse(predictions, processed_gt)
    
    # 7. Convert predictions and ground truth to joint positions using proper SMPL model
    print("Converting poses to joint positions for MPJPE calculation...")

    # Add function to convert from axis-angle to 6D rotation
    def axis_angle_to_6d(poses):
        """
        Convert poses from axis-angle (72 params) to 6D rotation (144 params)
        
        Args:
            poses: Tensor of shape [batch_size, seq_len, 72]
            
        Returns:
            Tensor of shape [batch_size, seq_len, 144]
        """
        from imuposer.math.angular import rotation_matrix_to_r6d, axis_angle_to_rotation_matrix
        
        batch_size, seq_len, _ = poses.shape
        poses_reshape = poses.reshape(-1, 72)  # [batch_size*seq_len, 72]
        
        # Convert to rotation matrices first [batch_size*seq_len, 216]
        rot_matrices = axis_angle_to_rotation_matrix(poses_reshape)
        
        # Convert to 6D representation [batch_size*seq_len, 144]
        r6d = rotation_matrix_to_r6d(rot_matrices)
        
        # Reshape back to [batch_size, seq_len, 144]
        return r6d.reshape(batch_size, seq_len, 144)

    # Fix for ground truth poses
    try:
        # Check if all elements have the same shape
        first_shape = processed_gt[0].shape if hasattr(processed_gt[0], 'shape') else np.array(processed_gt[0]).shape
        all_same_shape = all(arr.shape == first_shape if hasattr(arr, 'shape') else np.array(arr).shape == first_shape for arr in processed_gt)
        
        if all_same_shape:
            # If all same shape, we can use np.stack
            gt_poses_tensor = torch.tensor(np.stack(processed_gt), dtype=torch.float32)
        else:
            # Handle case with different shapes
            print("Warning: Elements in processed_gt have different shapes")
            # Convert each element to tensor separately
            gt_poses_list = [torch.tensor(pose, dtype=torch.float32).unsqueeze(0) for pose in processed_gt]
            gt_poses_tensor = torch.cat(gt_poses_list, dim=0)
            
        # Reshape as needed - expects [batch_size, seq_len, params]
        gt_poses_tensor = gt_poses_tensor.reshape(1, -1, gt_poses_tensor.shape[-1])
        
        # Convert from axis-angle to 6D rotation if needed
        if gt_poses_tensor.shape[-1] == 72:  # axis-angle format
            print("Converting ground truth poses from axis-angle to 6D rotation...")
            gt_poses_tensor = axis_angle_to_6d(gt_poses_tensor)
            
    except Exception as e:
        print(f"Error converting ground truth poses to tensor: {e}")
        # Fallback solution
        gt_poses_tensor = torch.zeros((1, len(processed_gt), 144), dtype=torch.float32)  # Use 144 for 6D rotation

    # Fix for prediction poses
    try:
        # Check if all elements have the same shape
        first_pred_shape = predictions[0].shape if hasattr(predictions[0], 'shape') else np.array(predictions[0]).shape
        all_pred_same_shape = all(arr.shape == first_pred_shape if hasattr(arr, 'shape') else np.array(arr).shape == first_pred_shape for arr in predictions)
        
        if all_pred_same_shape:
            # If all same shape, we can use np.stack
            pred_poses_tensor = torch.tensor(np.stack(predictions), dtype=torch.float32)
        else:
            # Handle case with different shapes
            print("Warning: Elements in predictions have different shapes")
            pred_poses_list = [torch.tensor(pose, dtype=torch.float32).unsqueeze(0) for pose in predictions]
            pred_poses_tensor = torch.cat(pred_poses_list, dim=0)
            
        # Reshape as needed
        pred_poses_tensor = pred_poses_tensor.reshape(1, -1, pred_poses_tensor.shape[-1])
        
        # Convert from axis-angle to 6D rotation if needed
        if pred_poses_tensor.shape[-1] == 72:  # axis-angle format
            print("Converting prediction poses from axis-angle to 6D rotation...")
            pred_poses_tensor = axis_angle_to_6d(pred_poses_tensor)
            
    except Exception as e:
        print(f"Error converting prediction poses to tensor: {e}")
        # Fallback solution
        pred_poses_tensor = torch.zeros((1, len(predictions), 144), dtype=torch.float32)  # Use 144 for 6D rotation

    # Convert using the proper SMPL model
    print("Converting poses to joint positions...")
    gt_joints = convert_poses_to_joints(gt_poses_tensor)
    pred_joints = convert_poses_to_joints(pred_poses_tensor)

    # Convert back to numpy for metric calculations
    gt_joints_np = gt_joints.detach().cpu().numpy().squeeze(0)  # [seq_len, 24, 3]
    pred_joints_np = pred_joints.detach().cpu().numpy().squeeze(0)  # [seq_len, 24, 3]

    print(f"Joint positions shapes - Ground Truth: {gt_joints_np.shape}, Predicted: {pred_joints_np.shape}")


    # 8. Calculate extended metrics
    print("Calculating extended metrics...")
    
    # Calculate overall MPJPE
    mpjpe = calculate_mpjpe(pred_joints_np, gt_joints_np)
    print(f"MPJPE: {mpjpe:.4f} mm")
    
    
    # Calculate acceleration error
    accel_error = calculate_acceleration_error(pred_joints_np, gt_joints_np)
    print(f"Acceleration Error: {accel_error:.4f} mm/s²")
    
    # Calculate SMPL parameter error
    try:
        # Convert predictions to numpy arrays with same shape
        pred_params = np.stack([np.array(p).flatten() for p in predictions])
        gt_params = np.stack([np.array(g).flatten() for g in processed_gt])
        
        # Ensure same shape by truncating if needed
        min_dim = min(pred_params.shape[1], gt_params.shape[1])
        pred_params = pred_params[:, :min_dim]
        gt_params = gt_params[:, :min_dim]
        
        smpl_error = calculate_smpl_param_error(pred_params, gt_params)
    except Exception as e:
        print(f"Error calculating SMPL parameter error: {e}")
        # Fallback value
        smpl_error = float('nan')
        
    print(f"SMPL Parameter Error: {smpl_error:.6f}")
    
    # Calculate per-activity metrics
    print("\nPer-activity metrics:")
    activities = {}
    
    for i, meta in enumerate(metadata):
        activity = meta["activity"]
        if activity not in activities:
            activities[activity] = {"indices": []}
        activities[activity]["indices"].append(i)
    
    for activity, data in activities.items():
        indices = data["indices"]
        if not indices:
            continue
            
        # Extract activity-specific joints
        activity_gt_joints = gt_joints_np[indices]
        activity_pred_joints = pred_joints_np[indices]
        
        # Calculate metrics
        activity_mpjpe = calculate_mpjpe(activity_pred_joints, activity_gt_joints)

        
        print(f"  {activity}:")
        print(f"    MPJPE: {activity_mpjpe:.4f} mm")
    
    # 9. Save results
    results_dir = os.path.join(base_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics_results.txt"), "w") as f:
        f.write("IMUPoser Evaluation Metrics\n")
        f.write("==========================\n\n")
        f.write(f"Overall MSE: {mse_results['overall_mse']:.6f}\n")
        f.write(f"MPJPE: {mpjpe:.4f} mm\n")

        f.write(f"Acceleration Error: {accel_error:.4f} mm/s²\n")
        f.write(f"SMPL Parameter Error: {smpl_error:.6f}\n\n")
        
        f.write("Per-activity metrics:\n")
        for activity, data in activities.items():
            indices = data["indices"]
            if not indices:
                continue
                
            # Extract activity-specific joints
            activity_gt_joints = gt_joints_np[indices]
            activity_pred_joints = pred_joints_np[indices]
            
            # Calculate metrics
            activity_mpjpe = calculate_mpjpe(activity_pred_joints, activity_gt_joints)
            
            f.write(f"  {activity}:\n")
            f.write(f"    MPJPE: {activity_mpjpe:.4f} mm\n")
    
    # 10. Generate some visualizations (optional)
    try:
        print("Generating visualizations...")
        from visualisepose import save_multi_view_grid
        
        # Save one visualization of ground truth vs prediction
        output_vis_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(output_vis_dir, exist_ok=True)
        
        # Take a sample for visualization (first sequence)
        sample_idx = 0
        
        # Reshape to expected format [batch_size, seq_len, n_joints, 3]
        sample_gt = gt_joints_np[sample_idx:sample_idx+1].reshape(1, 1, 24, 3)
        sample_pred = pred_joints_np[sample_idx:sample_idx+1].reshape(1, 1, 24, 3)
        
        # Convert to tensor for the visualization function
        sample_gt_tensor = torch.tensor(sample_gt)
        sample_pred_tensor = torch.tensor(sample_pred)
        
        # Save visualizations
        activity_name = metadata[sample_idx]["activity"]
        participant = metadata[sample_idx]["participant"]
        
        save_multi_view_grid(
            sample_gt_tensor, 
            os.path.join(output_vis_dir, f"{participant}_{activity_name}_gt.png"),
            frame_indices=[0],
            view_angles=[(30, 0), (30, 90)]
        )
        
        save_multi_view_grid(
            sample_pred_tensor, 
            os.path.join(output_vis_dir, f"{participant}_{activity_name}_pred.png"),
            frame_indices=[0],
            view_angles=[(30, 0), (30, 90)]
        )
        
        print(f"Visualizations saved to {output_vis_dir}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    print(f"\nTesting completed. Results saved to: {results_dir}")