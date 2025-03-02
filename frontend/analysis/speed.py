import torch 
import numpy as np

def process_movement_speed(predictions):
    """Process predictions to get movement speeds for all joints"""
    try:
        # Print what type we're working with
        print("Predictions type:", type(predictions))
        
        # Handle tensor input directly
        if isinstance(predictions, torch.Tensor):
            print("Working with tensor directly")
            joints = predictions
        # Handle dictionary with 'joints' key
        elif isinstance(predictions, dict) and 'joints' in predictions:
            print("Working with dictionary with 'joints' key")
            joints = predictions['joints']
        else:
            raise ValueError(f"Unsupported predictions type: {type(predictions)}")
        
        # Print shape information
        if isinstance(joints, torch.Tensor):
            print("Joints shape:", joints.shape)
        
        # Remove batch dimension and reshape
        if isinstance(joints, torch.Tensor):
            if len(joints.shape) > 2:  # If there's a batch dimension
                joints = joints.squeeze(0).cpu().numpy()
            else:
                joints = joints.cpu().numpy()
            print("After squeeze - shape:", joints.shape)
        
        n_frames = joints.shape[0]
        
        # Determine the correct reshape based on the shape
        if len(joints.shape) == 2:
            # For flat tensor, reshape based on the number of elements
            if joints.shape[1] == 72:  # 24 joints * 3 coordinates
                joints = joints.reshape(n_frames, 24, 3)
                n_joints = 24
            elif joints.shape[1] == 144:  # Probably includes rotations as well as positions
                # Use only the first 72 values (positions) and reshape
                joints = joints[:, :72].reshape(n_frames, 24, 3)
                n_joints = 24
            else:
                # Try to infer the number of joints
                n_joints = joints.shape[1] // 3
                joints = joints.reshape(n_frames, n_joints, 3)
        else:
            n_joints = joints.shape[1]
        
        print("After reshape - shape:", joints.shape)
        print(f"Analyzing speeds for {n_joints} joints")
        
        # SMPL joint names (if using standard SMPL model)
        joint_names = [
            "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", 
            "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", 
            "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
        ]
        
        # Adjust joint names if we have fewer joints
        if n_joints < len(joint_names):
            joint_names = joint_names[:n_joints]
        elif n_joints > len(joint_names):
            # Add generic names for extra joints
            joint_names.extend([f"joint_{i}" for i in range(len(joint_names), n_joints)])
        
        # Calculate velocities
        fps = 30  # Video framerate
        dt = 1/fps
        scale_factor = 100  # Convert to cm/s
        
        # Initialize speeds dictionary with all joint names
        speeds = {}
        for i, name in enumerate(joint_names):
            speeds[f"{name}_speed"] = []
        
        # Calculate speeds for each joint
        for i in range(1, n_frames):
            for j in range(n_joints):
                # Calculate displacement between frames
                disp = joints[i, j] - joints[i-1, j]
                # Calculate speed and add to list
                speeds[f"{joint_names[j]}_speed"].append(np.linalg.norm(disp) * scale_factor / dt)
        
        # Add a zero at the start for each joint (since we can't calculate speed for the first frame)
        for key in speeds:
            speeds[key].insert(0, 0)
        
        # Calculate statistics
        stats = {}
        for joint, velocities in speeds.items():
            avg_vel = np.mean([v for v in velocities if v > 0.1] or [0])
            max_vel = np.max(velocities) if velocities else 0
            min_vel = np.min([v for v in velocities if v > 0.1] or [0])
            
            stats[joint] = {
                'average': avg_vel,
                'max': max_vel,
                'min': min_vel
            }
        
        return speeds, stats
        
    except Exception as e:
        print(f"Error in process_movement_speed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Return empty data on error with at least a few basic joints
        basic_joints = ["pelvis", "left_hand", "right_hand", "left_foot", "right_foot"]
        empty_speeds = {f"{j}_speed": [0] for j in basic_joints}
        empty_stats = {k: {'average': 0, 'max': 0, 'min': 0} for k in empty_speeds.keys()}
        return empty_speeds, empty_stats