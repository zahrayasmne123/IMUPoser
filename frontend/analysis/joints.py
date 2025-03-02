import torch
import numpy as np

def calculate_angle(p1, p2, p3):
    """Calculate angle between three 3D points"""
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Check for zero vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    cosine = np.dot(v1, v2) / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    
    return np.degrees(angle)

def process_joint_angles(predictions):
    """Process predictions to get joint angles for all relevant joints"""
    try:
        # Print what type we're working with
        print("Joint angles - predictions type:", type(predictions))
        
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
        
        # Convert to numpy if it's a torch tensor
        if isinstance(joints, torch.Tensor):
            # Print shape information
            print("Joint tensor shape:", joints.shape)
            
            # Remove batch dimension if present
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
            elif joints.shape[1] == 144:  # Probably includes rotations as well as positions
                # Use only the first 72 values (positions) and reshape
                joints = joints[:, :72].reshape(n_frames, 24, 3)
            else:
                # Try to infer the number of joints
                n_joints = joints.shape[1] // 3
                joints = joints.reshape(n_frames, n_joints, 3)
        
        print("After reshape - shape:", joints.shape)
        n_joints = joints.shape[1]
        print(f"Analyzing angles for {n_joints} joints")
        
        # Define joint connections for common angles in SMPL model
        # Format: (joint_name, [parent_idx, joint_idx, child_idx])
        joint_connections = {
            # Legs
            'left_knee': ['left_hip', 'left_knee', 'left_ankle'],
            'right_knee': ['right_hip', 'right_knee', 'right_ankle'],
            'left_hip': ['spine1', 'left_hip', 'left_knee'],
            'right_hip': ['spine1', 'right_hip', 'right_knee'],
            'left_ankle': ['left_knee', 'left_ankle', 'left_foot'],
            'right_ankle': ['right_knee', 'right_ankle', 'right_foot'],
            
            # Arms
            'left_elbow': ['left_shoulder', 'left_elbow', 'left_wrist'],
            'right_elbow': ['right_shoulder', 'right_elbow', 'right_wrist'],
            'left_shoulder': ['left_collar', 'left_shoulder', 'left_elbow'],
            'right_shoulder': ['right_collar', 'right_shoulder', 'right_elbow'],
            'left_wrist': ['left_elbow', 'left_wrist', 'left_hand'],
            'right_wrist': ['right_elbow', 'right_wrist', 'right_hand'],
            
            # Spine
            'lower_back': ['pelvis', 'spine1', 'spine2'],
            'mid_back': ['spine1', 'spine2', 'spine3'],
            'upper_back': ['spine2', 'spine3', 'neck'],
            'neck': ['spine3', 'neck', 'head']
        }
        
        # SMPL joint names and indices
        smpl_joints = {
            "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3, 
            "left_knee": 4, "right_knee": 5, "spine2": 6, "left_ankle": 7, 
            "right_ankle": 8, "spine3": 9, "left_foot": 10, "right_foot": 11, 
            "neck": 12, "left_collar": 13, "right_collar": 14, "head": 15, 
            "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18, 
            "right_elbow": 19, "left_wrist": 20, "right_wrist": 21, 
            "left_hand": 22, "right_hand": 23
        }
        
        # Adjust if we have fewer joints
        if n_joints < 24:
            # Create a mapping with available joints
            ratio = n_joints / 24
            available_joints = {}
            for name, idx in smpl_joints.items():
                new_idx = min(int(idx * ratio), n_joints - 1)
                available_joints[name] = new_idx
            
            # Update joint_connections to only include available connections
            valid_connections = {}
            for angle_name, (parent, joint, child) in joint_connections.items():
                if parent in available_joints and joint in available_joints and child in available_joints:
                    valid_connections[angle_name] = [
                        available_joints[parent],
                        available_joints[joint],
                        available_joints[child]
                    ]
            
            joint_indices = available_joints
            angle_connections = valid_connections
        else:
            # Use standard indices
            angle_connections = {
                name: [smpl_joints[p], smpl_joints[j], smpl_joints[c]] 
                for name, (p, j, c) in joint_connections.items()
            }
            joint_indices = smpl_joints
        
        # Calculate angles for each connection
        angles = {}
        for angle_name, (parent_idx, joint_idx, child_idx) in angle_connections.items():
            angles[angle_name] = []
            for frame in joints:
                angle = calculate_angle(
                    frame[parent_idx],
                    frame[joint_idx],
                    frame[child_idx]
                )
                angles[angle_name].append(angle)
        
        return angles
        
    except Exception as e:
        print(f"Error in process_joint_angles: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Return basic angles on error
        empty_angles = {
            'left_knee': [0],
            'right_knee': [0],
            'left_elbow': [0],
            'right_elbow': [0]
        }
        return empty_angles