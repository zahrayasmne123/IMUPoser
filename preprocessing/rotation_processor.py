import pandas as pd
import numpy as np

"""Rotation Processor: 
1. Process_acceleration_data: Converts measurements from g units to m/s² and applies a scale factor 
2. Rotation_matrix_single_conversion: Uses gyroscope measurements to calculate the 3D rotation matrix needed to IMUPoser inpt
3. Rotation_matrix_df_conversion: Takes data input, processes accelerometer and gyroscope data using previous functions and 
returns complete dataframe with correct units and column names"""

## process_acceleration_data: Converts measurements from g units to m/s² and applies a scale factor 
## Outputs new columns with converted units and removes the original column
def process_acceleration_data(df):
    scale_factor = 30
    conversion_factor = 9.81  # Convert g to m/s^2

    # Rename column
    for df_column in ['x-axis (g)', 'y-axis (g)', 'z-axis (g)']:
        if df_column in df.columns:
            new_column_name = df_column.replace("(g)", "(m/s^2)")
            
            # Handle empty values 
            df[df_column] = df[df_column].replace([np.inf, -np.inf], 0).fillna(0)
            
            try:
                # Compute conversion
                df[new_column_name] = df[df_column] * conversion_factor * scale_factor 
            except Exception as e:
                print(f"Error processing {df_column}: {e}")
            
            df.drop(columns=[df_column], inplace=True)  # Remove old column

    return df

#Compute angular displacements for each axis over the time step (1/60)
def calc_theta(w):
    ANGULAR_THRESHOLD = 1e-3  # degrees per second
    delta_t=1/60
    absolute_rotation = abs(w * delta_t)
    if absolute_rotation < ANGULAR_THRESHOLD:
        return 0.0
    return np.clip(w * delta_t, -np.pi/2, np.pi/2)
    

## rotation_matrix_conversion: Sequential step by step calculation of a 3D rotation matrix from gyroscope measurements.
# Function takes gyroscope data, converts them to radians, computes the incremental rotation for time step of 1/60
# and builds a rotation matrix using the roll-pitch-yaw convention. 
def rotation_matrix_singular_conversion(gyro_x, gyro_y, gyro_z):
    EPSILON = 1e-6 #threshold to determine if an angle is small enough to skip
    
    #1. convert input to radians
    try:
        wx_rad = np.radians(gyro_x)
        wy_rad = np.radians(gyro_y)
        wz_rad = np.radians(gyro_z)
    except Exception:
        wx_rad, wy_rad, wz_rad = 0.0, 0.0, 0.0
    
    #2. angular displacement for each axis over time step (1/60)
    theta_x = calc_theta(wx_rad)
    theta_y = calc_theta(wy_rad)
    theta_z = calc_theta(wz_rad)
    
    #Compute sine and cosine values for each axis
    # If angle is < EPSILON, use (1.0, 0.0) instead of calculating
    cos_x, sin_x = (1.0, 0.0) if abs(theta_x) < EPSILON else (np.cos(theta_x), np.sin(theta_x))
    cos_y, sin_y = (1.0, 0.0) if abs(theta_y) < EPSILON else (np.cos(theta_y), np.sin(theta_y))
    cos_z, sin_z = (1.0, 0.0) if abs(theta_z) < EPSILON else (np.cos(theta_z), np.sin(theta_z))
    
    #3. Construct individual rotation matrices for rotations around the X, Y , and Z axes:
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
    
    #4. Combine rotations
    try:
        # Multiply matrices in order R_z @ R_y @ R_x (z-y-x convention)
        final_rotation = R_z @ R_y @ R_x 
        
        # Check if result is orthogonal (must be orgononal to be a valid rotation matrix)
        if not np.allclose(np.linalg.det(final_rotation), 1.0, atol=1e-3):
            return np.eye(3) 
        
        return final_rotation
    except Exception:
        return np.eye(3) #identity matrix used as fallback (no rotation)
    
#rotation_matrix_df_conversion: Takes input accelerometer and gyroscope dataframes and applies gyroscope rotation matrices covnersion
#Returns a list of processed dfs 
def rotation_matrix_df_conversion(dataframes):
    processed_dfs = []

    for i, df in enumerate(dataframes):
        df = process_acceleration_data(df.copy())  #first process acceleration data

        print("Computing rotation matrices...")
        matrices_array = []
        for _, row in df.iterrows():
            gyro_x, gyro_y, gyro_z = row['x-axis (deg/s)'], row['y-axis (deg/s)'], row['z-axis (deg/s)']
            R = rotation_matrix_singular_conversion(gyro_x, gyro_y, gyro_z) #apply gyroscope conversion
            matrices_array.append(R.flatten()) #flattens the 3x3 rotation matrix into a 1D array of 9 elements

        #  Make a new df  from the list of flattened matrices
        rotation_df = pd.DataFrame(matrices_array, 
                                   columns=[f"R{i}{j}" for i in range(3) for j in range(3)])
        # Concatenate back to original form
        final_df = pd.concat([df[['timestamp', 'x-axis (m/s^2)', 'y-axis (m/s^2)', 'z-axis (m/s^2)']], 
                               rotation_df], axis=1)
        
        print(f"Output shape: {final_df.shape}")
        processed_dfs.append(final_df)

    return processed_dfs