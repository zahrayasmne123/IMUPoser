import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import seaborn as sns
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

# Define sensor combinations to test
SENSOR_COMBINATIONS = [
    # Single sensors
    {"name": "Left Wrist Only", "indices": [0]},
    {"name": "Right Wrist Only", "indices": [1]},
    {"name": "Left Pocket Only", "indices": [2]},
    {"name": "Right Pocket Only", "indices": [3]},
    {"name": "Head Only", "indices": [4]},
    
    # Two sensors
    {"name": "Both Wrists", "indices": [0, 1]},
    {"name": "Left Wrist + Left Pocket", "indices": [0, 2]},
    {"name": "Left Wrist + Right Pocket", "indices": [0, 3]},
    {"name": "Left Wrist + Head", "indices": [0, 4]},
    {"name": "Right Wrist + Left Pocket", "indices": [1, 2]},
    {"name": "Right Wrist + Right Pocket", "indices": [1, 3]},
    {"name": "Right Wrist + Head", "indices": [1, 4]},
    {"name": "Both Pockets", "indices": [2, 3]},
    {"name": "Left Pocket + Head", "indices": [2, 4]},
    {"name": "Right Pocket + Head", "indices": [3, 4]},
    
    # Three sensors
    {"name": "Both Wrists + Left Pocket", "indices": [0, 1, 2]},
    {"name": "Both Wrists + Right Pocket", "indices": [0, 1, 3]},
    {"name": "Both Wrists + Head", "indices": [0, 1, 4]},
    {"name": "Left Wrist + Both Pockets", "indices": [0, 2, 3]},
    {"name": "Right Wrist + Both Pockets", "indices": [1, 2, 3]},
    {"name": "Left Wrist + Left Pocket + Head", "indices": [0, 2, 4]},
    {"name": "Right Wrist + Right Pocket + Head", "indices": [1, 3, 4]},
    {"name": "Both Pockets + Head", "indices": [2, 3, 4]},
    
    # Four sensors
    {"name": "Both Wrists + Both Pockets", "indices": [0, 1, 2, 3]},
    {"name": "Both Wrists + Left Pocket + Head", "indices": [0, 1, 2, 4]},
    {"name": "Both Wrists + Right Pocket + Head", "indices": [0, 1, 3, 4]},
    {"name": "Left Wrist + Both Pockets + Head", "indices": [0, 2, 3, 4]},
    {"name": "Right Wrist + Both Pockets + Head", "indices": [1, 2, 3, 4]},
    
    # All sensors
    {"name": "All Sensors", "indices": [0, 1, 2, 3, 4]},
]

def load_test_data(participant_ids=None, activities=None):
    """Load test data from the IMUPoser dataset"""
    if participant_ids is None:
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
            
            if activities is not None and not any(act in activity_name for act in activities):
                continue
            
            try:
                with open(activity_file, "rb") as f:
                    data = pkl.load(f)
                
                # Extract IMU data and ground truth pose
                imu_data = data["imu"]
                pose_gt = data["pose"]
                
                # Extract translation and betas if available
                trans_gt = data.get("trans", None)
                betas_gt = data.get("betas", None)
                
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

def create_masked_input(imu_data, active_indices):
    """
    Create masked IMU input with only specified sensors active
    
    Args:
        imu_data: Original IMU data [seq_len, 60]
        active_indices: List of active sensor indices (0-4)
        
    Returns:
        Masked IMU data with inactive sensors set to 0
    """
    # Create a copy of the data to avoid modifying the original
    masked_data = np.zeros_like(imu_data)
    
    # For each active sensor, copy its data
    for idx in active_indices:
        # Acceleration data (3 values per sensor)
        acc_start = idx * 3
        acc_end = acc_start + 3
        masked_data[:, acc_start:acc_end] = imu_data[:, acc_start:acc_end]
        
        # Orientation data (9 values per sensor, as 3x3 matrices)
        ori_start = 15 + (idx * 9)  # 15 = 5 sensors * 3 acc values
        ori_end = ori_start + 9
        masked_data[:, ori_start:ori_end] = imu_data[:, ori_start:ori_end]
    
    return masked_data

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
        List of predicted poses
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
                
                pred_pose = pred_pose.cpu().numpy()
            
            # Remove batch dimension if present
            if len(pred_pose.shape) > 1 and pred_pose.shape[0] == 1:
                pred_pose = pred_pose.squeeze(0)
                
            predictions.append(pred_pose)
    
    return predictions

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
    if len(predicted_joints) < 3 or len(ground_truth_joints) < 3:
        print("Warning: Sequence too short for acceleration error calculation")
        return float('nan')
        
    # Calculate acceleration for predicted joints (2nd derivative)
    pred_acc = (predicted_joints[2:] - 2 * predicted_joints[1:-1] + predicted_joints[:-2]) / (dt**2)
    
    # Calculate acceleration for ground truth joints
    gt_acc = (ground_truth_joints[2:] - 2 * ground_truth_joints[1:-1] + ground_truth_joints[:-2]) / (dt**2)
    
    # Calculate acceleration error for each joint in each frame
    acc_error = np.sqrt(np.sum((pred_acc - gt_acc)**2, axis=2))
    
    # Average across joints and frames
    mean_acc_error = np.mean(acc_error)
    
    return mean_acc_error

def evaluate_sensor_combinations(model, test_data, convert_poses_to_joints_fn=None, combinations=SENSOR_COMBINATIONS, window_size=125, stride=25, device="cpu", max_examples=5):
    """
    Evaluate model performance across different sensor combinations
    
    Args:
        model: Your trained IMUPoser model
        test_data: Dictionary containing test data
        convert_poses_to_joints_fn: Function to convert SMPL poses to joint positions
        combinations: List of sensor combinations to evaluate
        window_size: Size of sliding window for input
        stride: Stride between windows
        device: Device to run inference on
        max_examples: Maximum number of sequences to evaluate
        
    Returns:
        DataFrame with results for each combination
    """
    results = []
    model.eval()
    
    for combo in tqdm(combinations, desc="Evaluating sensor combinations"):
        combo_name = combo["name"]
        active_indices = combo["indices"]
        print(f"\nEvaluating combination: {combo_name}")
        
        all_mse = []  # Fall back to MSE if joint conversion fails
        
        for i, (imu, pose) in enumerate(zip(test_data["imu_inputs"][:max_examples], 
                                           test_data["pose_gt"][:max_examples])):
            activity = test_data["metadata"][i]["activity"]
            print(f"  Processing sequence {i+1}/{min(max_examples, len(test_data['imu_inputs']))}: {activity}")
            
            # Create masked input with only specified sensors active
            masked_imu = create_masked_input(imu, active_indices)
            
            # Create sliding windows
            imu_windows = []
            pose_targets = []
            
            for j in range(0, len(masked_imu) - window_size + 1, stride):
                imu_window = masked_imu[j:j+window_size]
                # Use middle frame as target
                middle_frame_idx = j + window_size // 2
                if middle_frame_idx < len(pose):
                    pose_target = pose[middle_frame_idx]
                    imu_windows.append(imu_window)
                    pose_targets.append(pose_target)
            
            if not imu_windows:
                print(f"    Warning: No valid windows for sequence {i+1}")
                continue
                
            print(f"    Running inference on {len(imu_windows)} windows")
            
            # Run inference
            predictions = run_inference(model, imu_windows, device=device)
            
            # Make sure prediction and target formats match
            # Convert any tensors to numpy arrays
            for k in range(len(pose_targets)):
                if isinstance(pose_targets[k], torch.Tensor):
                    pose_targets[k] = pose_targets[k].cpu().numpy()
            
            # Calculate MSE on pose parameters directly
            try:
                # Check if shapes match
                batch_mse = []
                for p, t in zip(predictions, pose_targets):
                    # Ensure both are numpy arrays with matching shapes
                    if isinstance(p, torch.Tensor):
                        p = p.cpu().numpy()
                    if isinstance(t, torch.Tensor):
                        t = t.cpu().numpy()
                    
                    # Ensure shapes match
                    min_dim = min(p.shape[0], t.shape[0])
                    p_trunc = p[:min_dim]
                    t_trunc = t[:min_dim]
                    
                    # Calculate MSE
                    mse = np.mean((p_trunc - t_trunc) ** 2)
                    batch_mse.append(mse)
                
                # Average MSE for this sequence
                seq_mse = np.mean(batch_mse)
                all_mse.append(seq_mse)
                print(f"    MSE (pose parameters): {seq_mse:.6f}")
            except Exception as e:
                print(f"    Error calculating MSE: {e}")
        
        # Store results using MSE (since MPJPE conversion failed)
        avg_mse = np.mean(all_mse) if all_mse else float('nan')
        
        results.append({
            "combination": combo_name,
            "num_sensors": len(active_indices),
            "mse": avg_mse,
            "active_sensors": ", ".join([["Left Wrist", "Right Wrist", "Left Pocket", "Right Pocket", "Head"][i] for i in active_indices])
        })
        
        print(f"  Average MSE: {avg_mse:.6f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def visualize_results(results_df, output_file="sensor_combination_analysis.png"):
    """
    Visualize the results comparing sensor combinations
    
    Args:
        results_df: DataFrame with results from evaluate_sensor_combinations
        output_file: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    # Set up the figure
    plt.figure(figsize=(15, 12))
    
    # Plot 1: MSE by number of sensors (boxplot)
    plt.subplot(2, 2, 1)
    sns.boxplot(x="num_sensors", y="mse", data=results_df)
    plt.title("MSE by Number of Sensors")
    plt.xlabel("Number of Sensors")
    plt.ylabel("Mean Squared Error")
    
    # Plot 2: MSE by number of sensors (line plot with mean and std)
    plt.subplot(2, 2, 2)
    grouped = results_df.groupby("num_sensors")["mse"].agg(["mean", "std"]).reset_index()
    plt.errorbar(grouped["num_sensors"], grouped["mean"], yerr=grouped["std"], marker="o")
    plt.title("MSE by Number of Sensors (Mean ± Std)")
    plt.xlabel("Number of Sensors")
    plt.ylabel("Mean Squared Error")
    
    # Plot 3: Top 5 best combinations
    plt.subplot(2, 2, 3)
    top_5 = results_df.sort_values("mse").head(5)
    sns.barplot(x="combination", y="mse", data=top_5)
    plt.title("Top 5 Sensor Combinations")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Squared Error")
    
    # Plot 4: Best combination for each number of sensors
    plt.subplot(2, 2, 4)
    best_per_count = results_df.loc[results_df.groupby("num_sensors")["mse"].idxmin()]
    sns.barplot(x="num_sensors", y="mse", data=best_per_count)
    plt.title("Best Combination per Sensor Count")
    plt.xlabel("Number of Sensors")
    plt.ylabel("Mean Squared Error")
    
    # Add combination labels to the bars
    for i, row in enumerate(best_per_count.itertuples()):
        plt.text(i, row.mse + 0.0001, row.combination, ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create detailed table of results
    detailed_results = results_df.sort_values("mse")
    print("\nDetailed Results (Sorted by MSE):")
    print(detailed_results[["combination", "num_sensors", "mse", "active_sensors"]])
    
    # Save the results to CSV
    csv_output = output_file.replace(".png", ".csv")
    detailed_results.to_csv(csv_output, index=False)
    print(f"Results saved to {csv_output}")
    
    # Return the name of the saved figure
    return output_file

def analyze_body_part_errors(model, test_data, convert_poses_to_joints_fn, body_parts=None, output_dir="body_part_analysis"):
    """
    Analyze errors for specific body parts across sensor combinations
    
    Args:
        model: Trained IMUPoser model
        test_data: Dictionary containing test data
        convert_poses_to_joints_fn: Function to convert SMPL poses to joint positions
        body_parts: Dictionary mapping body part names to joint indices
        output_dir: Directory to save results
    """
    if body_parts is None:
        # Default mapping of body parts to SMPL joint indices
        body_parts = {
            "Head": [15],
            "Left Arm": [13, 16, 18, 20, 22],  # Shoulder, elbow, wrist, etc.
            "Right Arm": [14, 17, 19, 21, 23],
            "Left Leg": [1, 4, 7, 10],  # Hip, knee, ankle, etc.
            "Right Leg": [2, 5, 8, 11],
            "Torso": [0, 3, 6, 9, 12]
        }
    
    # Select a few key sensor combinations
    key_combinations = [
        {"name": "Left Wrist Only", "indices": [0]},
        {"name": "Right Wrist Only", "indices": [1]},
        {"name": "Left Pocket Only", "indices": [2]},
        {"name": "Right Pocket Only", "indices": [3]},
        {"name": "Head Only", "indices": [4]},
        {"name": "Both Wrists", "indices": [0, 1]},
        {"name": "Both Pockets", "indices": [2, 3]},
        {"name": "Left Wrist + Left Pocket", "indices": [0, 2]},
        {"name": "Right Wrist + Right Pocket", "indices": [1, 3]},
        {"name": "Both Wrists + Head", "indices": [0, 1, 4]},
        {"name": "All Sensors", "indices": [0, 1, 2, 3, 4]}
    ]
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results structure
    body_part_results = []
    
    # Maximum number of sequences to analyze
    max_examples = 3
    
    for combo in tqdm(key_combinations, desc="Analyzing body parts"):
        combo_name = combo["name"]
        active_indices = combo["indices"]
        print(f"\nAnalyzing combination: {combo_name}")
        
        for i, (imu, pose) in enumerate(zip(test_data["imu_inputs"][:max_examples], 
                                           test_data["pose_gt"][:max_examples])):
            activity = test_data["metadata"][i]["activity"]
            print(f"  Processing sequence {i+1}/{min(max_examples, len(test_data['imu_inputs']))}: {activity}")
            
            # Create masked input with only specified sensors active
            masked_imu = create_masked_input(imu, active_indices)
            
            # Create sliding windows
            imu_windows = []
            pose_targets = []
            
            # Use a larger stride for efficiency
            for j in range(0, len(masked_imu) - 125 + 1, 50):
                imu_window = masked_imu[j:j+125]
                pose_target = pose[j + 62]  # Middle frame
                imu_windows.append(imu_window)
                pose_targets.append(pose_target)
            
            # Run inference
            predictions = run_inference(model, imu_windows, device="cpu")
            
            # Convert poses to joint positions
            try:
                # Stack predictions and targets
                pred_poses = np.stack(predictions)
                gt_poses = np.stack(pose_targets)
                
                # Convert to tensors
                pred_poses_tensor = torch.tensor(pred_poses, dtype=torch.float32).unsqueeze(0)
                gt_poses_tensor = torch.tensor(gt_poses, dtype=torch.float32).unsqueeze(0)
                
                # Convert to joint positions
                pred_joints = convert_poses_to_joints_fn(pred_poses_tensor)
                gt_joints = convert_poses_to_joints_fn(gt_poses_tensor)
                
                # Convert to numpy
                pred_joints_np = pred_joints.cpu().numpy().squeeze(0)  # [seq_len, 24, 3]
                gt_joints_np = gt_joints.cpu().numpy().squeeze(0)  # [seq_len, 24, 3]
                
                # Calculate error for each body part
                for body_part, joint_indices in body_parts.items():
                    joint_errors = []
                    
                    for joint_idx in joint_indices:
                        # Calculate Euclidean distance for this joint
                        joint_error = np.sqrt(np.sum((pred_joints_np[:, joint_idx, :] - gt_joints_np[:, joint_idx, :]) ** 2, axis=1))
                        joint_errors.append(joint_error)
                    
                    # Average error across all joints in this body part
                    body_part_error = np.mean(np.concatenate(joint_errors))
                    
                    # Store the result
                    body_part_results.append({
                        "combination": combo_name,
                        "num_sensors": len(active_indices),
                        "body_part": body_part,
                        "error": body_part_error,
                        "participant": test_data["metadata"][i]["participant"],
                        "activity": activity
                    })
                    
                    print(f"    {body_part} Error: {body_part_error:.4f} mm")
            except Exception as e:
                print(f"    Error in body part analysis: {e}")
    
    
        # Convert to DataFrame
    results_df = pd.DataFrame(body_part_results)
    
    # Save the results
    results_df.to_csv(os.path.join(output_dir, "body_part_errors.csv"), index=False)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Body part errors by number of sensors
    plt.subplot(2, 2, 1)
    sns.boxplot(x="num_sensors", y="error", data=results_df)
    plt.title("Body Part Error by Number of Sensors")
    plt.xlabel("Number of Sensors")
    plt.ylabel("Error (mm)")
    
    # Plot 2: Body part errors by body part
    plt.subplot(2, 2, 2)
    sns.boxplot(x="body_part", y="error", data=results_df)
    plt.title("Error by Body Part")
    plt.xlabel("Body Part")
    plt.ylabel("Error (mm)")
    plt.xticks(rotation=45)
    
    # Plot 3: Heatmap of body part errors by combination
    plt.subplot(2, 1, 2)
    pivot_table = results_df.pivot_table(values="error", index="combination", columns="body_part", aggfunc="mean")
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis_r")
    plt.title("Body Part Errors by Sensor Combination")
    plt.xlabel("Body Part")
    plt.ylabel("Sensor Combination")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "body_part_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create per-body part analysis
    for body_part in body_parts.keys():
        plt.figure(figsize=(10, 6))
        part_data = results_df[results_df["body_part"] == body_part]
        
        # Sort combinations by number of sensors
        part_data = part_data.sort_values(["num_sensors", "error"])
        
        sns.barplot(x="combination", y="error", data=part_data)
        plt.title(f"{body_part} Error by Sensor Combination")
        plt.xlabel("Sensor Combination")
        plt.ylabel("Error (mm)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"{body_part.lower().replace(' ', '_')}_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"Body part analysis saved to {output_dir}")
    return results_df

# Main execution
if __name__ == "__main__":
    print("Starting IMUPoser sensor combination analysis...")
    
    # Create results directory
    results_dir = os.path.join(base_dir, "sensor_analysis_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load your trained model
    checkpoint_path = os.path.join(base_dir, "checkpoint.ckpt")
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path)
    
    # 2. Load test data with specific activities
    print("Loading test data...")
    test_data = load_test_data(
        participant_ids=["P1", "P2", "P3"],  # Using 3 participants for analysis
        activities=["Walking", "ArmRaises", "Boxing", "Pushups"]  # Choose diverse activities
    )
    
    # 3. Evaluate sensor combinations - don't use the joint conversion
    print("Evaluating sensor combinations...")
    results_df = evaluate_sensor_combinations(
        model=model, 
        test_data=test_data,
        convert_poses_to_joints_fn=None,  # Don't use the conversion function
        max_examples=3  # Limit to 3 sequences for faster evaluation
    )
    
    # 4. Visualize results
    figure_path = visualize_results(
        results_df,
        output_file=os.path.join(results_dir, "sensor_combination_analysis.png")
    )
    print(f"Results visualization saved to: {figure_path}")
    
    print(f"Analysis complete. Results saved to {results_dir}")
   
