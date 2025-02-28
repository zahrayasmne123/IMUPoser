import torch 
import numpy as np

def process_movement_speed(predictions_dict):
    """Process predictions dictionary to get movement speeds"""
    try:
        # Get the joints data with proper error checking
        if not isinstance(predictions_dict, dict):
            raise ValueError(f"Expected dictionary, got {type(predictions_dict)}")
            
        if 'joints' not in predictions_dict:
            raise ValueError(f"No 'joints' key in dictionary. Available keys: {list(predictions_dict.keys())}")
            
        joints = predictions_dict['joints']
        print("Joints type:", type(joints))
        print("Joints shape:", joints.shape if hasattr(joints, 'shape') else "No shape attribute")
        
        # Remove batch dimension and reshape
        if isinstance(joints, torch.Tensor):
            joints = joints.squeeze(0).cpu().numpy()  # Remove batch dimension
            print("After squeeze - shape:", joints.shape)
        
        n_frames = joints.shape[0]
        joints = joints.reshape(n_frames, 24, 3)  # Reshape to (frames, joints, xyz)
        print("After reshape - shape:", joints.shape)
        
        
        # Calculate velocities
        fps = 30  # From your video stats
        dt = 1/fps
        scale_factor = 100  # Convert to cm/s
        
        speeds = {
            'right_hand_speed': [],
            'left_hand_speed': [],
            'right_foot_speed': [],
            'left_foot_speed': [],
            'center_speed': []
        }
        
        # Joint indices
        LEFT_WRIST = 19
        RIGHT_WRIST = 20
        LEFT_ANKLE = 7
        RIGHT_ANKLE = 8
        PELVIS = 0
        
        # Calculate speeds for each joint
        for i in range(1, n_frames):
            # Calculate displacement between frames
            right_hand_disp = joints[i, RIGHT_WRIST] - joints[i-1, RIGHT_WRIST]
            left_hand_disp = joints[i, LEFT_WRIST] - joints[i-1, LEFT_WRIST]
            right_foot_disp = joints[i, RIGHT_ANKLE] - joints[i-1, RIGHT_ANKLE]
            left_foot_disp = joints[i, LEFT_ANKLE] - joints[i-1, LEFT_ANKLE]
            center_disp = joints[i, PELVIS] - joints[i-1, PELVIS]
            
            # Print some sample displacements for debugging
            if i == 1:
                print("\nSample displacements (frame 1):")
                print("Right hand displacement:", right_hand_disp)
                print("Calculated speed:", np.linalg.norm(right_hand_disp) * scale_factor / dt)
            
            # Calculate speeds with scaling
            speeds['right_hand_speed'].append(np.linalg.norm(right_hand_disp) * scale_factor / dt)
            speeds['left_hand_speed'].append(np.linalg.norm(left_hand_disp) * scale_factor / dt)
            speeds['right_foot_speed'].append(np.linalg.norm(right_foot_disp) * scale_factor / dt)
            speeds['left_foot_speed'].append(np.linalg.norm(left_foot_disp) * scale_factor / dt)
            speeds['center_speed'].append(np.linalg.norm(center_disp) * scale_factor / dt)
        
        # Add a zero at the start since we can't calculate speed for the first frame
        for key in speeds:
            speeds[key].insert(0, 0)
        
        # Calculate statistics
        stats = {}
        for joint, velocities in speeds.items():
            non_zero_velocities = [v for v in velocities if v > 0.1]
            if non_zero_velocities:
                stats[joint] = {
                    'average_speed': np.mean(non_zero_velocities),
                    'peak_speed': np.max(velocities),
                    'min_speed': np.min(non_zero_velocities)
                }
            else:
                stats[joint] = {
                    'average_speed': 0,
                    'peak_speed': 0,
                    'min_speed': 0
                }
        
        return speeds, stats
        
    except Exception as e:
        print(f"Error in process_movement_speed: {str(e)}")
        raise