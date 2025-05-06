import pandas as pd
import numpy as np

def robust_normalize_acceleration(df, convert_to_zero=True):
    scale_factor = 30
    conversion_factor = 9.81  # Convert g to m/s^2

    for axis in ['x-axis (g)', 'y-axis (g)', 'z-axis (g)']:
        if axis in df.columns:
            new_column_name = axis.replace("(g)", "(m/s^2)")
            
            # Handle NaN and Inf values
            if convert_to_zero:
                df[axis] = df[axis].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Compute with error handling
            try:
                df[new_column_name] = df[axis] * conversion_factor * scale_factor
            except Exception as e:
                print(f"Error processing {axis}: {e}")
                # Fallback to zeros if computation fails
                df[new_column_name] = 0
            
            df.drop(columns=[axis], inplace=True)  # Drop old column

    return df

def robust_compute_rotation_matrix(gyro_x, gyro_y, gyro_z, delta_t=1/60):
    # Safely convert inputs
    def safe_value(x, default=0.0):
        try:
            return float(x) if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default
    
    # Constants
    EPSILON = 1e-6
    ANGULAR_THRESHOLD = 1e-3  # Degrees per second
    
    # Process inputs
    wx = safe_value(gyro_x)
    wy = safe_value(gyro_y)
    wz = safe_value(gyro_z)
    
    # Convert to radians
    try:
        wx_rad = np.radians(wx)
        wy_rad = np.radians(wy)
        wz_rad = np.radians(wz)
    except Exception:
        wx_rad, wy_rad, wz_rad = 0.0, 0.0, 0.0
    
    # Compute angular displacements with thresholds
    def compute_theta(w):
        abs_rot = abs(w * delta_t)
        if abs_rot < ANGULAR_THRESHOLD:
            return 0.0
        return np.clip(w * delta_t, -np.pi/2, np.pi/2)
    
    theta_x = compute_theta(wx_rad)
    theta_y = compute_theta(wy_rad)
    theta_z = compute_theta(wz_rad)
    
    # Compute trig values
    cos_x, sin_x = (1.0, 0.0) if abs(theta_x) < EPSILON else (np.cos(theta_x), np.sin(theta_x))
    cos_y, sin_y = (1.0, 0.0) if abs(theta_y) < EPSILON else (np.cos(theta_y), np.sin(theta_y))
    cos_z, sin_z = (1.0, 0.0) if abs(theta_z) < EPSILON else (np.cos(theta_z), np.sin(theta_z))
    
    # Construct rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    
    R_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    
    R_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    
    # Combine rotations
    try:
        final_rotation = R_z @ R_y @ R_x
        
        # Single orthogonality check
        if not (np.allclose(np.linalg.det(final_rotation), 1.0, atol=1e-3)):
            return np.eye(3)
        
        return final_rotation
    except Exception:
        return np.eye(3)
    
def robust_rotation_matrices_dataframes(dataframes, fps=60):
    print(f"Starting robust processing of {len(dataframes)} dataframes at {fps} FPS")
    delta_t = 1 / fps
    processed_dfs = []

    for i, df in enumerate(dataframes):
        print(f"\nProcessing dataframe {i+1}/{len(dataframes)}")
        print(f"Input shape: {df.shape}")

        required_columns = ['x-axis (g)', 'y-axis (g)', 'z-axis (g)', 
                            'x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping dataframe {i+1}: Missing required columns")
            continue

        print("Normalizing acceleration values...")
        df = robust_normalize_acceleration(df.copy())

        print("Computing rotation matrices...")
        rotation_matrices = []
        for _, row in df.iterrows():
            gyro_x, gyro_y, gyro_z = row['x-axis (deg/s)'], row['y-axis (deg/s)'], row['z-axis (deg/s)']
            R = robust_compute_rotation_matrix(gyro_x, gyro_y, gyro_z, delta_t)
            rotation_matrices.append(R.flatten())

        rotation_df = pd.DataFrame(rotation_matrices, 
                                   columns=[f"R{i}{j}" for i in range(3) for j in range(3)])
        result_df = pd.concat([df[['timestamp', 'x-axis (m/s^2)', 'y-axis (m/s^2)', 'z-axis (m/s^2)']], 
                               rotation_df], axis=1)
        
        print(f"Output shape: {result_df.shape}")
        processed_dfs.append(result_df)

    print(f"\nCompleted robust processing of {len(processed_dfs)} dataframes")
    return processed_dfs