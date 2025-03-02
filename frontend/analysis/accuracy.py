import torch 
import numpy as np

def process_pose_accuracy(predictions):
    """Process predictions to get comprehensive pose accuracy metrics"""
    try:
        # Print what type we're working with
        print("Pose accuracy - predictions type:", type(predictions))
        
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
            print("Joints tensor shape:", joints.shape)
            
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
        
        # Initialize metrics
        metrics = {
            # Overall metrics
            'smoothness': np.zeros(n_frames),
            'symmetry_score': np.zeros(n_frames),
            'posture_score': np.zeros(n_frames),
            
            # Regional smoothness metrics
            'upper_body_smoothness': np.zeros(n_frames),
            'lower_body_smoothness': np.zeros(n_frames),
            'torso_smoothness': np.zeros(n_frames),
            
            # Regional symmetry metrics
            'arm_symmetry': np.zeros(n_frames),
            'leg_symmetry': np.zeros(n_frames),
            
            # Posture metrics
            'head_tilt': np.zeros(n_frames),
            'vertical_alignment': np.zeros(n_frames),
        }
        
        # Define symmetric joint pairs for different body regions
        joint_pairs = {
            "arms": [(13, 14), (16, 17), (18, 19), (20, 21), (22, 23)],  # collar, shoulder, elbow, wrist, hand
            "legs": [(1, 2), (4, 5), (7, 8), (10, 11)]  # hip, knee, ankle, foot
        }
        
        # Define spine joints for posture analysis
        spine_joints = [0, 3, 6, 9, 12, 15]  # pelvis, spine1, spine2, spine3, neck, head
        
        # Adjust joint indices if necessary
        if n_joints < 24:
            # Scale joint indices based on available joints
            ratio = n_joints / 24
            
            # Scale joint pairs
            scaled_pairs = {}
            for region, pairs in joint_pairs.items():
                scaled_pairs[region] = []
                for left, right in pairs:
                    scaled_left = min(int(left * ratio), n_joints - 1)
                    scaled_right = min(int(right * ratio), n_joints - 1)
                    if scaled_left != scaled_right:  # Ensure they're different joints
                        scaled_pairs[region].append((scaled_left, scaled_right))
            joint_pairs = scaled_pairs
            
            # Scale spine joints
            spine_joints = [min(int(j * ratio), n_joints - 1) for j in spine_joints]
            # Ensure no duplicates in spine_joints
            spine_joints = sorted(list(set(spine_joints)))
        
        # Define joint groups for regional smoothness
        upper_body_joints = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # neck and above, shoulders, arms
        lower_body_joints = [1, 2, 4, 5, 7, 8, 10, 11]  # hips, legs
        torso_joints = [0, 3, 6, 9]  # pelvis, spine
        
        # Scale joint groups if necessary
        if n_joints < 24:
            ratio = n_joints / 24
            upper_body_joints = [min(int(j * ratio), n_joints - 1) for j in upper_body_joints]
            lower_body_joints = [min(int(j * ratio), n_joints - 1) for j in lower_body_joints]
            torso_joints = [min(int(j * ratio), n_joints - 1) for j in torso_joints]
            
            # Remove duplicates
            upper_body_joints = sorted(list(set(upper_body_joints)))
            lower_body_joints = sorted(list(set(lower_body_joints)))
            torso_joints = sorted(list(set(torso_joints)))
        
        # Calculate metrics for each frame
        for i in range(n_frames):
            # Overall smoothness (jerk)
            if i >= 2:
                pos_curr = joints[i]
                pos_prev = joints[i-1]
                pos_prev2 = joints[i-2]
                jerk = np.mean(np.abs(pos_curr - 2*pos_prev + pos_prev2))
                metrics['smoothness'][i] = 1 / (1 + jerk)
                
                # Regional smoothness
                if upper_body_joints:
                    upper_jerk = np.mean(np.abs(pos_curr[upper_body_joints] - 2*pos_prev[upper_body_joints] + pos_prev2[upper_body_joints]))
                    metrics['upper_body_smoothness'][i] = 1 / (1 + upper_jerk)
                
                if lower_body_joints:
                    lower_jerk = np.mean(np.abs(pos_curr[lower_body_joints] - 2*pos_prev[lower_body_joints] + pos_prev2[lower_body_joints]))
                    metrics['lower_body_smoothness'][i] = 1 / (1 + lower_jerk)
                
                if torso_joints:
                    torso_jerk = np.mean(np.abs(pos_curr[torso_joints] - 2*pos_prev[torso_joints] + pos_prev2[torso_joints]))
                    metrics['torso_smoothness'][i] = 1 / (1 + torso_jerk)
            
            elif i > 0:
                # Copy from previous frame
                metrics['smoothness'][i] = metrics['smoothness'][i-1]
                metrics['upper_body_smoothness'][i] = metrics['upper_body_smoothness'][i-1]
                metrics['lower_body_smoothness'][i] = metrics['lower_body_smoothness'][i-1]
                metrics['torso_smoothness'][i] = metrics['torso_smoothness'][i-1]
            
            # Left-right symmetry
            # Arms symmetry
            if 'arms' in joint_pairs and joint_pairs['arms']:
                arm_symmetry_scores = []
                for left, right in joint_pairs['arms']:
                    left_pos = joints[i, left]
                    right_pos = joints[i, right]
                    # Calculate symmetry using mirrored coordinates
                    mirrored_right = np.array([-1, 1, 1]) * right_pos
                    dist = np.linalg.norm(left_pos - mirrored_right)
                    max_dist = np.linalg.norm(left_pos) + np.linalg.norm(right_pos)
                    if max_dist > 0:
                        symmetry = 1 - (dist / (2 * max_dist))
                        arm_symmetry_scores.append(symmetry)
                
                if arm_symmetry_scores:
                    metrics['arm_symmetry'][i] = np.mean(arm_symmetry_scores)
            
            # Legs symmetry
            if 'legs' in joint_pairs and joint_pairs['legs']:
                leg_symmetry_scores = []
                for left, right in joint_pairs['legs']:
                    left_pos = joints[i, left]
                    right_pos = joints[i, right]
                    # Calculate symmetry using mirrored coordinates
                    mirrored_right = np.array([-1, 1, 1]) * right_pos
                    dist = np.linalg.norm(left_pos - mirrored_right)
                    max_dist = np.linalg.norm(left_pos) + np.linalg.norm(right_pos)
                    if max_dist > 0:
                        symmetry = 1 - (dist / (2 * max_dist))
                        leg_symmetry_scores.append(symmetry)
                
                if leg_symmetry_scores:
                    metrics['leg_symmetry'][i] = np.mean(leg_symmetry_scores)
            
            # Overall symmetry (average of arm and leg symmetry)
            arm_sym = metrics['arm_symmetry'][i]
            leg_sym = metrics['leg_symmetry'][i]
            if arm_sym > 0 and leg_sym > 0:
                metrics['symmetry_score'][i] = (arm_sym + leg_sym) / 2
            elif arm_sym > 0:
                metrics['symmetry_score'][i] = arm_sym
            elif leg_sym > 0:
                metrics['symmetry_score'][i] = leg_sym
            
            # Posture score (spine alignment)
            if len(spine_joints) > 1:
                # Extract spine vectors
                spine_vectors = []
                for j in range(len(spine_joints)-1):
                    spine_vectors.append(joints[i, spine_joints[j+1]] - joints[i, spine_joints[j]])
                
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
                        # Perfect posture would be a straight line (π radians between vectors)
                        metrics['posture_score'][i] = 1 - np.mean(np.abs(np.array(spine_angles) - np.pi)) / np.pi
            
            # Head tilt (if we have enough joints)
            if len(spine_joints) > 4:  # Need at least spine3, neck, head
                neck_idx = spine_joints[-2]  # Second to last in spine chain
                head_idx = spine_joints[-1]  # Last in spine chain
                
                # Get head orientation vector
                head_vec = joints[i, head_idx] - joints[i, neck_idx]
                
                # Calculate angle with vertical axis [0,1,0]
                vertical = np.array([0, 1, 0])
                head_vec_norm = np.linalg.norm(head_vec)
                
                if head_vec_norm > 0:
                    cos_angle = np.dot(head_vec, vertical) / head_vec_norm
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    # Convert to a score (0-1) where 1 is perfectly vertical
                    metrics['head_tilt'][i] = 1 - angle / np.pi
            
            # Vertical alignment (how aligned the spine is with vertical axis)
            if len(spine_joints) > 1:
                first_spine = spine_joints[0]  # Usually pelvis
                last_spine = spine_joints[-1]  # Usually head or neck
                
                # Vector from pelvis to head
                spine_vec = joints[i, last_spine] - joints[i, first_spine]
                
                # Calculate alignment with vertical
                vertical = np.array([0, 1, 0])
                spine_vec_norm = np.linalg.norm(spine_vec)
                
                if spine_vec_norm > 0:
                    cos_angle = np.dot(spine_vec, vertical) / spine_vec_norm
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    # Convert to a score (0-1) where 1 is perfectly vertical
                    metrics['vertical_alignment'][i] = 1 - angle / np.pi
        
        # Convert metrics to lists for pandas
        metrics_list = {key: list(values) for key, values in metrics.items()}
        
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
        
    except Exception as e:
        print(f"Error in process_pose_accuracy: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Return empty metrics on error
        basic_metrics = {
            'smoothness': [0] * 10,
            'symmetry_score': [0] * 10,
            'posture_score': [0] * 10
        }
        basic_stats = {
            'smoothness': {'average': 0, 'max': 0, 'min': 0},
            'symmetry_score': {'average': 0, 'max': 0, 'min': 0},
            'posture_score': {'average': 0, 'max': 0, 'min': 0}
        }
        return basic_metrics, basic_stats