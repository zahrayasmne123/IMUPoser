import torch
import numpy as np

def calculate_angle(p1, p2, p3):
    """Calculate angle between three 3D points"""
    v1 = p2 - p1
    v2 = p3 - p2
    
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    
    return np.degrees(angle)

def process_joint_angles(predictions_dict):
    """Process predictions dictionary to get joint angles"""
    # Get the joints data
    joints = predictions_dict['joints']
    
    # Convert to numpy if it's a torch tensor
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    
    # Print shape for debugging
    print("Joint data shape:", joints.shape)
    
    # The shape should be (frames, joints, 3)
    # If it's flattened, we need to reshape it correctly
    if len(joints.shape) == 2:
        n_frames = joints.shape[0]
        joints = joints.reshape(n_frames, -1, 3)
    
    angles = {
        'left_knee': [],
        'right_knee': [],
        'left_elbow': [],
        'right_elbow': []
    }
    
    for frame in joints:
        # Calculate knee angles
        angles['left_knee'].append(calculate_angle(
            frame[1],   # left hip
            frame[4],   # left knee
            frame[7]    # left ankle
        ))
        
        angles['right_knee'].append(calculate_angle(
            frame[2],   # right hip
            frame[5],   # right knee
            frame[8]    # right ankle
        ))
        
        # Calculate elbow angles
        angles['left_elbow'].append(calculate_angle(
            frame[13],  # left shoulder
            frame[16],  # left elbow
            frame[19]   # left wrist
        ))
        
        angles['right_elbow'].append(calculate_angle(
            frame[14],  # right shoulder
            frame[17],  # right elbow
            frame[20]   # right wrist
        ))
    
    return angles
