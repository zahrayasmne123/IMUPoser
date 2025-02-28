import pandas as pd
import numpy as np

def validate_dataframe(df):
    """Perform comprehensive data validation before processing."""
    # Check for NaN values
    nan_counts = df.isna().sum()
    print("NaN counts before processing:")
    print(nan_counts)
    
    # Check for infinite values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    print("\nInfinite value counts:")
    print(inf_counts)
    
    # Basic statistical validation
    print("\nBasic statistical validation:")
    for col in df.select_dtypes(include=[np.number]).columns:
        stats = df[col].describe()
        print(f"\n{col} statistics:")
        print(f"Min: {stats['min']}")
        print(f"Max: {stats['max']}")
        print(f"Mean: {stats['mean']}")
        print(f"Standard Deviation: {stats['std']}")

def robust_normalize_acceleration(df, convert_to_zero=True):
    """
    Robust acceleration normalization with error handling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        convert_to_zero (bool): If True, replace NaN/Inf with 0
    
    Returns:
        pd.DataFrame: Normalized dataframe
    """
    # Validate input first
    validate_dataframe(df)
    
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
    """
    Enhanced robust rotation matrix computation with advanced numerical stability techniques.
    
    Args:
        gyro_x, gyro_y, gyro_z (float): Gyroscope readings in degrees/sec
        delta_t (float): Time delta, default 1/60 second
    
    Returns:
        np.ndarray: Rotation matrix with improved numerical stability
    """
    # Comprehensive input validation and preprocessing
    def safe_value(x, default=0.0):
        """Safely handle numeric inputs."""
        try:
            return float(x) if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default
    
    # Safely convert inputs
    wx = safe_value(gyro_x)
    wy = safe_value(gyro_y)
    wz = safe_value(gyro_z)
    
    # Numerical stability threshold
    EPSILON = 1e-6
    ANGULAR_THRESHOLD = 1e-3  # Degrees per second
    
    # Convert to radians with additional safety
    try:
        wx_rad = np.radians(wx)
        wy_rad = np.radians(wy)
        wz_rad = np.radians(wz)
    except Exception:
        wx_rad, wy_rad, wz_rad = 0.0, 0.0, 0.0
    
    # Compute angular displacement with extreme care
    def safe_angular_displacement(angular_velocity, time_delta):
        """
        Compute angular displacement with multiple numerical stability techniques.
        
        Handles:
        - Very small rotations
        - Potential floating-point instabilities
        - Extreme rotation values
        """
        # Absolute rotation magnitude
        abs_rotation = abs(angular_velocity * time_delta)
        
        # If rotation is below threshold, return near-identity transformation
        if abs_rotation < ANGULAR_THRESHOLD:
            return 0.0
        
        # Clip extreme values
        clipped_rotation = np.clip(
            angular_velocity * time_delta, 
            -np.pi/2, 
            np.pi/2
        )
        
        return clipped_rotation
    
    # Compute safe angular displacements
    theta_x = safe_angular_displacement(wx_rad, delta_t)
    theta_y = safe_angular_displacement(wy_rad, delta_t)
    theta_z = safe_angular_displacement(wz_rad, delta_t)
    
    # Compute trigonometric values with numerical stability
    def safe_trig(theta):
        """Compute cos and sin with numerical stability."""
        if abs(theta) < EPSILON:
            return 1.0, 0.0
        return np.cos(theta), np.sin(theta)
    
    cos_x, sin_x = safe_trig(theta_x)
    cos_y, sin_y = safe_trig(theta_y)
    cos_z, sin_z = safe_trig(theta_z)
    
    # Construct rotation matrices with added numerical checks
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
    
    # Combine rotations (ZYX order) with final numerical stability check
    try:
        final_rotation = R_z @ R_y @ R_x
        
        # Ensure orthogonality and determinant close to 1
        if not (np.allclose(np.linalg.det(final_rotation), 1.0, atol=1e-3) and 
                np.allclose(final_rotation.T @ final_rotation, np.eye(3), atol=1e-3)):
            return np.eye(3)
        
        return final_rotation
    except Exception:
        return np.eye(3)
    
def robust_rotation_matrices_dataframes(dataframes, fps=60):
    """
    Process dataframes with comprehensive error handling.
    
    Args:
        dataframes (list): List of input dataframes
        fps (int): Frames per second
    
    Returns:
        list: Processed dataframes
    """
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

