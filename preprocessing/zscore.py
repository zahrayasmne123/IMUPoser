import numpy as np 

# Applies Z-score anomaly detection to identify and remove outliers in IMU sensor data.
def filter_outliers_zscore(df, columns, threshold=3.0):    
    filtered_df = df.copy()
    
    # Store outlier statistics
    outlier_stats = {
        'total_datapoints': len(df) * len(columns),
        'total_outliers': 0,
        'outliers_per_column': {},
        'outlier_percentage': 0.0
    }
    
    # Process each column
    for column in columns:
        if column not in filtered_df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame, skipping.")
            continue
            
        # Calculate z-scores
        mean = filtered_df[column].mean()
        std = filtered_df[column].std()
        z_scores = np.abs((filtered_df[column] - mean) / std)
        
        # Identify outliers
        outliers = z_scores > threshold
        outlier_count = outliers.sum()
        
        # Store statistics
        outlier_stats['outliers_per_column'][column] = outlier_count
        outlier_stats['total_outliers'] += outlier_count
        
        if outlier_count > 0:
            print(f"Found {outlier_count} outliers in column '{column}'")
            
            # Replace outliers with NaN
            filtered_df.loc[outliers, column] = np.nan
            filtered_df[column] = filtered_df[column].interpolate(method='linear')
            
            # Check for any remaining NaN values at edges
            if filtered_df[column].isna().any():
                filtered_df[column] = filtered_df[column].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate overall percentage
    if outlier_stats['total_datapoints'] > 0:
        outlier_stats['outlier_percentage'] = (outlier_stats['total_outliers'] / 
                                              outlier_stats['total_datapoints']) * 100
    
    print("Z-score filtering summary:")
    print(f"  Total data points analyzed: {outlier_stats['total_datapoints']}")
    print(f"  Total outliers detected: {outlier_stats['total_outliers']} ({outlier_stats['outlier_percentage']:.4f}%)")
    
    return filtered_df, outlier_stats


def apply_zscore_filtering_to_sensors(sensor_dfs, df_names=None, threshold=3.0):
    """
    Apply Z-score filtering to a list of sensor dataframes.
    
    Args:
        sensor_dfs: List of sensor dataframes (can contain None values)
        df_names: Names of corresponding dataframes
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Tuple of (filtered_dfs, stats) containing filtered dataframes and outlier statistics
    """
    if df_names is None:
        df_names = ['phone', 'earbuds', 'left_watch', 'right_watch']
    
    filtered_dfs = []
    all_stats = {}
    
    for i, df in enumerate(sensor_dfs):
        if df is None:
            filtered_dfs.append(None)
            continue
            
        device_name = df_names[i]
        print(f"\nProcessing {device_name} data for outliers...")
        
        # Identify columns to check based on data type
        accel_columns = [col for col in df.columns if '(g)' in col or '(m/s^2)' in col]
        gyro_columns = [col for col in df.columns if '(deg/s)' in col]
        
        # Apply filtering to accelerometer data
        if accel_columns:
            print("Filtering accelerometer data...")
            df, accel_stats = filter_outliers_zscore(df, accel_columns, threshold)
            all_stats[f"{device_name}_accel"] = accel_stats
            
        # Apply filtering to gyroscope data
        if gyro_columns:
            print("Filtering gyroscope data...")
            df, gyro_stats = filter_outliers_zscore(df, gyro_columns, threshold)
            all_stats[f"{device_name}_gyro"] = gyro_stats
            
        filtered_dfs.append(df)
        
    return filtered_dfs, all_stats