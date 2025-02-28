import torch 
import numpy as np

def process_pose_accuracy(predictions_dict):
    """Process predictions dictionary to get pose accuracy metrics"""
    joints = predictions_dict['joints']
    
    # Remove batch dimension and reshape
    if isinstance(joints, torch.Tensor):
        joints = joints.squeeze(0).cpu().numpy()
    
    n_frames = joints.shape[0]
    joints = joints.reshape(n_frames, 24, 3)  # Reshape to (frames, joints, xyz)
    
    # Initialize arrays for metrics with the correct length
    metrics = {
        'smoothness': np.zeros(n_frames),
        'symmetry_score': np.zeros(n_frames),
        'posture_score': np.zeros(n_frames)
    }
    
    # Calculate metrics for each frame
    for i in range(n_frames):
        # Motion smoothness
        if i >= 2:
            pos_curr = joints[i]
            pos_prev = joints[i-1]
            pos_prev2 = joints[i-2]
            jerk = np.mean(np.abs(pos_curr - 2*pos_prev + pos_prev2))
            metrics['smoothness'][i] = 1 / (1 + jerk)
        elif i > 0:
            metrics['smoothness'][i] = metrics['smoothness'][i-1]
        
        # Left-right symmetry
        joint_pairs = [
            (13, 14),  # shoulders
            (16, 17),  # elbows
            (19, 20),  # wrists
            (4, 5),    # knees
            (7, 8)     # ankles
        ]
        symmetry_scores = []
        for left, right in joint_pairs:
            left_pos = joints[i, left]
            right_pos = joints[i, right]
            symmetry = 1 - (np.linalg.norm(left_pos - np.array([-1, 1, 1])*right_pos) / 2)
            symmetry_scores.append(symmetry)
        metrics['symmetry_score'][i] = np.mean(symmetry_scores)
        
        # Posture score
        spine_joints = [0, 3, 6, 9]  # pelvis to head
        spine_vectors = np.diff(joints[i, spine_joints], axis=0)
        if len(spine_vectors) > 1:
            spine_angles = []
            for j in range(len(spine_vectors)-1):
                v1_norm = np.linalg.norm(spine_vectors[j])
                v2_norm = np.linalg.norm(spine_vectors[j+1])
                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(spine_vectors[j], spine_vectors[j+1]) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    spine_angles.append(angle)
            if spine_angles:
                metrics['posture_score'][i] = 1 - np.mean(np.abs(np.array(spine_angles) - np.pi)) / np.pi
    
    # Convert to dictionary of lists for pandas
    metrics_list = {
        key: list(values) for key, values in metrics.items()
    }
    
    # Calculate statistics
    stats = {}
    for metric, values in metrics_list.items():
        non_zero_values = [v for v in values if v > 0]
        if non_zero_values:
            stats[metric] = {
                'average': np.mean(non_zero_values),
                'max': np.max(values),
                'min': np.min(non_zero_values)
            }
        else:
            stats[metric] = {
                'average': 0,
                'max': 0,
                'min': 0
            }
    
    return metrics_list, stats