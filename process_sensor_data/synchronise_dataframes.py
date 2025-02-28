import pandas as pd
import numpy as np
from scipy import interpolate

def robust_synchronise_dataframes(dfs, device_names):
    """
    Robust synchronization of dataframes with improved NaN handling
    
    Parameters:
    - dfs: List of input dataframes
    - device_names: Corresponding names of devices
    
    Returns:
    - List of synchronized dataframes
    """
    processed_dfs = []
    
    # Standardize timestamp formats
    for df in dfs:
        if 'timestamp' in df.columns:
            # Extract time components from the various formats and create standard format
            def standardize_timestamp(ts):
                if 'T' in ts:  # ISO format
                    time_part = ts.split('T')[1]
                    return time_part.replace('.', ':').split('+')[0]
                else:  # Phone/earbud format
                    return ts
            df['timestamp'] = df['timestamp'].apply(standardize_timestamp)
    
    # Convert timestamps to datetime
    for df in dfs:
        try:
            # Add a date part to make valid datetime
            df['timestamp'] = pd.to_datetime(
                '2025-02-25 ' + df['timestamp'].str.replace(':', '.', n=3),
                format='%Y-%m-%d %H.%M.%S.%f',
                errors='coerce'
            )
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            print(f"Sample timestamps: {df['timestamp'].head().tolist()}")
    
    # Find common time range
    start_time = max(df['timestamp'].min() for df in dfs)
    end_time = min(df['timestamp'].max() for df in dfs)
    start_time = start_time.round('ms')
    
    print("\nCommon time range:")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")
    print(f"Duration: {(end_time - start_time).total_seconds():.2f} seconds")
    
    # Generate target timestamps at 30 FPS
    fps = 30
    duration_seconds = (end_time - start_time).total_seconds()
    num_frames = int(duration_seconds * fps)
    
    target_timestamps = [
        start_time + pd.Timedelta(seconds=i/fps) 
        for i in range(num_frames)
    ]
    
    # Process each dataframe
    for df, name in zip(dfs, device_names):
        print(f"\nProcessing {name}...")
        
        # Filter out rows outside the common time range
        df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
        
        # Prepare resampled data
        resampled_data = {'timestamp': target_timestamps}
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Use original timestamp for interpolation
            x = (df_filtered['timestamp'] - start_time).dt.total_seconds()
            y = df_filtered[col].values
            
            # Create interpolation function with more robust method
            f = interpolate.interp1d(
                x, 
                y, 
                kind='linear',  # Changed from 'cubic' to 'linear' for more stability
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Generate new x values based on target timestamps
            x_new = (pd.Series(target_timestamps) - start_time).dt.total_seconds()
            
            try:
                # Interpolate the values
                interpolated_values = f(x_new)
                
                # Additional NaN handling
                interpolated_values = np.nan_to_num(
                    interpolated_values, 
                    nan=np.nanmean(interpolated_values)  # Replace NaNs with mean
                )
                
                resampled_data[col] = interpolated_values
            
            except Exception as e:
                print(f"Error interpolating {col} for {name}: {e}")
                # Fallback: use original column with padding/truncating
                interpolated_values = np.pad(
                    y, 
                    (0, num_frames - len(y)), 
                    mode='constant', 
                    constant_values=np.mean(y)
                )[:num_frames]
                resampled_data[col] = interpolated_values
        
        # Create resampled dataframe
        df_resampled = pd.DataFrame(resampled_data)
        
        # Scale acceleration values (optional, adjust as needed)
        scale_factor = 1/30
        acc_columns = [col for col in df_resampled.columns if 'axis (m/s^2)' in col]
        for col in acc_columns:
            df_resampled[col] *= scale_factor
        
        # Convert timestamp back to string format
        df_resampled['timestamp'] = df_resampled['timestamp'].dt.strftime('%H:%M:%S:%f').str[:-3]
        
        processed_dfs.append(df_resampled)
    
    # Validate synchronization
    validate_synchronization(processed_dfs, device_names)
    
    return processed_dfs

def validate_synchronization(dfs, device_names):
    """
    Validate synchronization of dataframes
    
    Parameters:
    - dfs: List of synchronized dataframes
    - device_names: Corresponding device names
    """
    print("\nValidation Results:")
    print("-" * 50)
    
    # Check row counts
    row_counts = [len(df) for df in dfs]
    print(f"Row counts: {dict(zip(device_names, row_counts))}")
    
    if len(set(row_counts)) != 1:
        print("❌ WARNING: Different number of rows detected!")
    else:
        print("✓ All devices have the same number of rows")
    
    # Check for NaNs
    for df, name in zip(dfs, device_names):
        nan_counts = df.isna().sum()
        print(f"\nNaN counts for {name}:")
        print(nan_counts[nan_counts > 0])
    
    # ... (rest of the original validation logic remains the same)

def scale_acceleration_for_mobileposer(df):
    """
    Scale acceleration values by 1/30 to match MobilePoser's expected range.
    
    Parameters:
    - df: Input dataframe
    
    Returns:
    - Scaled dataframe
    """
    scale_factor = 1/30
    acc_columns = [col for col in df.columns if 'axis (m/s^2)' in col]
    for col in acc_columns:
        df[col] = df[col] * scale_factor
    return df

# Example usage
# device_names = ['Phone', 'Earbud', 'Left Watch', 'Right Watch']
# dfs = [phone_df, earbud_df, left_watch_df, right_watch_df]
# synchronized_dfs = robust_synchronise_dataframes(dfs, device_names)