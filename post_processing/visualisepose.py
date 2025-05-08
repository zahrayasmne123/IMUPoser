import torch

import matplotlib.pyplot as plt
from pathlib import Path
import os
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import axis_angle_to_rotation_matrix, r6d_to_rotation_matrix
from imuposer.config import BASE_DIR
from .visualisation_helpers import *  # noqa: F403

"""VISUALISE POSE: 
1. Prepare Joints Data: Selects the first sequence of joints and  if multiple  exist it converts the data 
   from tensors to arrays
2. Save Multi View Grid: Creates a grid of skeleton visualisations with customisable frames and angles whilst ensuring 
   scaling across each of the views. These are concetenated into a singular image. 
3. Save Individual Frames: Creates individual frames for animation at intervals of 5, saving frames 
4. Convert Poses to Joints: Loads the SMPL human body model and converts pose parameters to joint positions,
   performing forward kinematics to get 3D joint positions. Can take in input sizes of 72, 144 and 216. Input size
   72 is for IMUPoser dataset analysis and 144 for main TEMPO analysis
5. Load Predictions: Loads 3d pose estimation prediction from a files and gives debugging info in terminal
6. Full Visualisation Pipeline: Builds on previous functions to laod data, convert to SMPL, create ulti view frames and gif animation
saved in the correct directories


"""




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
 


def save_grid_viewer(
    joints_data, output_path="pose_grid.png", frame_indices=None, view_angles=None
):
    joints = prepare_joints_data(joints_data)     # Prepare joint data
    if frame_indices is None: # default frame indices is evenly spaced
        total_frames = len(joints)
        frame_indices = [int(i * total_frames / 6) for i in range(6)]

    #default view angles  are front an side views
    if view_angles is None:
        view_angles = [
            (30, 0),  # Front view
            (30, 90),  # Side view 
        ]

    #  axis limits limitations 
    axis_limits = calculate_axis_limits(joints)  # noqa: F405

    # Create a multi view  grid for displaying skeleton
    n_rows = len(frame_indices)
    n_cols = len(view_angles)
    skeleton_graph = plt.figure(figsize=(n_cols * 8, n_rows * 8))

    # Loop through frames and views
    for i, frame_idx in enumerate(frame_indices):
        joint_pos = joints[frame_idx]

        for j, (elev, azim) in enumerate(view_angles):
            # Create subplot
            ax = skeleton_graph.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            
            # Plot skeleton
            plot_skeleton_frame(  # noqa: F405
                ax, 
                joint_pos, 
                axis_limits, 
                vertical_viewing_angle=elev, 
                horizontal_rotation=azim, 
                graph_title=f"Frame {frame_idx} - View ({elev}°, {azim}°)"
            )

    # Save graph
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(skeleton_graph)

    print(f"Saved multi-view grid to {output_path}")
    return output_path


def save_individual_frames( joints_data, output_dir="pose_frames_dots", frame_interval=5):
    # Make output directory if not already existing
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    joints = prepare_joints_data(joints_data)
    axis_limits = calculate_axis_limits(joints)  # noqa: F405
    elev, azim = 30, 45  # Set view angle

    # Save frames
    for i in range(0, len(joints), frame_interval):
        joint_pos = joints[i]

        # Create graph
        skeleton_graph = plt.figure(figsize=(10, 10))
        ax = skeleton_graph.add_subplot(111, projection="3d")

        # Plot skeleton
        plot_skeleton_frame(  # noqa: F405
            ax, 
            joint_pos, 
            axis_limits, 
            vertical_viewing_angle=elev, 
            horizontal_rotation=azim, 
            graph_title=f"Frame {i}"
        )
        # Save graph
        plt.tight_layout()
        output_file = output_dir / f"frame_{i:04d}.png"
        plt.savefig(output_file, dpi=150)
        plt.close(skeleton_graph)

        if i % 50 == 0:
            print(f"Saved frame {i}/{len(joints)}")
    return output_dir




def convert_poses_to_joints(
    model_predictions,
    smpl_model_path=os.path.join(BASE_DIR, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
):

    body_model = ParametricModel(str(smpl_model_path), device=torch.device("cpu")) #initialse smpl model
    batch_size, sequence_length, num_parameters = model_predictions.shape # Get batch size and sequence length
    pose_prediction_reshape = model_predictions.reshape(-1, num_parameters) # Flatten the batch and sequence dimensions
    
    # Determine input format and convert if needed
    if num_parameters == 72:  # axis-angle format used for IMUPOser dataset
        print("Converting axis-angle rotations to rotation matrices...")
        rotation_matrix = axis_angle_to_rotation_matrix(pose_prediction_reshape)
        rotation_matrix_joints = rotation_matrix.reshape(-1, 24, 3, 3) # reshape to group by joints
        poses_matrices = rotation_matrix_joints.reshape(-1, 24, 9) #flatten the last two dimensions
        
        poses_matrices = poses_matrices.reshape(-1, 216) #flatten to [batch_size*seq_len, 216]
        
    elif num_parameters == 144:  # 6D representation for SMPL representation
        print("Converting 6D rotations to rotation matrices...")
        pose_estimations_6d = pose_prediction_reshape.reshape(-1, 24, 6) #Reshape to 24 joints x 6 parameters
        rotation_matrix = torch.zeros(pose_estimations_6d.shape[0], 24, 3, 3, device=pose_estimations_6d.device)
        for j in range(24):
            rotation_matrix[:, j] = r6d_to_rotation_matrix(pose_estimations_6d[:, j])
        
        # flatten
        poses_matrices = rotation_matrix.reshape(-1, 216)
        
    elif num_parameters == 216:  # already matrices
        poses_matrices = pose_prediction_reshape
    else:
        raise ValueError("Unexpected pose parameter size. Expected 72, 144, or 216.")

    print("Running SMPL forward kinematics...")
    result = body_model.forward_kinematics(pose=poses_matrices)  # Get joint positions from SMPL
    
    if len(result) == 2:
        _, joints = result
    else:
        _, joints, _ = result

    # Reshape back to [batch_size, seq_len, num_joints, 3]
    joints = joints.reshape(batch_size, sequence_length, -1, 3)

    print(f"Joint positions computed with shape: {joints.shape}")
    return joints



def load_predictions(predictions_path):
    predictions_path = Path(predictions_path)
    print(f"Loading predictions from: {predictions_path}") #debugging
    predictions = torch.load(predictions_path, map_location="cpu")
    print(f"Predictions shape: {predictions.shape}") #debugging
    return predictions

def full_visualisation_pipeline():
    predictions = load_predictions(os.path.join(BASE_DIR, "rawdata/processed/predictions.pt"))
    joints = convert_poses_to_joints(
        predictions, os.path.join(BASE_DIR, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    )

    frame_indices = [0, 100, 200, 300, 400, 500] if joints.shape[1] >= 500 else None
    save_grid_viewer(joints, os.path.join(BASE_DIR, "output/pose_grid_dots.png"), frame_indices)


    frames_dir = save_individual_frames(joints, os.path.join(BASE_DIR, "output/pose_frames_dots"), 5)
    create_gif(frames_dir, os.path.join(BASE_DIR, "output/output_frames.gif"))  # noqa: F405

    print("visualisation successfully completed!")
