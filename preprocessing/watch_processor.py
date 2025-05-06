import pandas as pd
import numpy as np

""" A class to handle watch sensor data alignment and adjustments.
    Provides methods to clean, standardize timestamp formats, and merge accelerometer
    and gyroscope data into a single file.
    """
class WatchSensorAligner:
    def __init__(self):
       self.timestamp_columns = ['timestamp (+0000)', 'timestamp', 'time (-00:00)']
       self.epoch_columns = ['epoc (ms)', 'epoch', 'epoch (ms)']
       self.expected_columns = {
           'accel': ['x-axis (g)', 'y-axis (g)', 'z-axis (g)'],
           'gyro': ['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
       }

    def process_timestamp(self, df):
        if df is None:
            return None
        
        timestamp_col = next((col for col in self.timestamp_columns if col in df.columns), None)
        
        if not timestamp_col:
            print(f"No timestamp column found. Available columns: {df.columns.tolist()}")
            return None
        
        try:
            # Simplified timestamp processing
            if timestamp_col == 'time (-00:00)':
                df['timestamp'] = df[timestamp_col]
            else:
                df['timestamp'] = df[timestamp_col].apply(
                    lambda ts: ts.split('T')[1].replace('.', ':').split('+')[0] if 'T' in str(ts) else ts
                )
            
            # Keep the original columns for now to debug
            cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
            return df[cols]
                
        except Exception as e:
            print(f"Error processing timestamps: {e}")
            return None

    def align_sensor_data(self, accel_df, gyro_df):
        if accel_df is None or gyro_df is None:
            print("Missing accelerometer or gyroscope data")
            return None
        
        try:
            # Make copies to avoid SettingWithCopyWarning
            accel_df = accel_df.copy()
            gyro_df = gyro_df.copy()
            
            # Get the time columns
            accel_time_col = next((col for col in self.timestamp_columns if col in accel_df.columns), None)
            gyro_time_col = next((col for col in self.timestamp_columns if col in gyro_df.columns), None)
            
            if accel_time_col is None or gyro_time_col is None:
                print("Missing time columns")
                return None
                
            # Create timestamp columns
            accel_df['timestamp'] = accel_df[accel_time_col]
            gyro_df['timestamp'] = gyro_df[gyro_time_col]
            
            # Create millisecond timestamps for matching
            accel_epoch_col = next((col for col in self.epoch_columns if col in accel_df.columns), None)
            gyro_epoch_col = next((col for col in self.epoch_columns if col in gyro_df.columns), None)
            
            if accel_epoch_col is not None and gyro_epoch_col is not None:
                # Use epoch milliseconds
                accel_df['timestamp_ms'] = accel_df[accel_epoch_col]
                gyro_df['timestamp_ms'] = gyro_df[gyro_epoch_col]
            else:
                # Extract milliseconds from timestamp strings
                accel_df['timestamp_ms'] = accel_df['timestamp'].apply(
                    lambda ts: int(ts.split('.')[-1]) if '.' in ts else 0
                )
                gyro_df['timestamp_ms'] = gyro_df['timestamp'].apply(
                    lambda ts: int(ts.split('.')[-1]) if '.' in ts else 0
                )
            
            # Create output dataframe
            merged_df = pd.DataFrame()
            merged_df['timestamp'] = accel_df['timestamp']
            
            # Add sensor columns with fallback to NaN
            for sensor_type, cols in self.expected_columns.items():
                source_df = accel_df if sensor_type == 'accel' else gyro_df
                
                for col in cols:
                    if col in source_df.columns:
                        if sensor_type == 'accel':
                            merged_df[col] = source_df[col]
                        else:
                            # Handle gyro data length differences
                            gyro_values = source_df[col].values
                            if len(gyro_values) != len(merged_df):
                                # Adjust gyro data length to match accel length
                                if len(gyro_values) > len(merged_df):
                                    gyro_values = gyro_values[:len(merged_df)]  # Truncate
                                else:
                                    # Pad with last value
                                    padding = np.full(len(merged_df) - len(gyro_values), gyro_values[-1])
                                    gyro_values = np.concatenate([gyro_values, padding])
                            merged_df[col] = gyro_values
                    else:
                        # Add NaN values for missing columns
                        merged_df[col] = np.nan
            
            return merged_df
            
        except Exception as e:
            print(f"Error merging sensor data: {e}")
            return None
            



#Bias correction for gyroscope data based on calibration values 
def correct_gyroscope_bias(aligned_dfs, df_names=None):
    """Apply bias correction to gyroscope data in aligned dataframes."""
    if df_names is None:
        df_names = ['phone', 'earbuds', 'left_watch', 'right_watch']
    
    # Store the calibration bias values
    device_biases = {
        'phone': {'x': 0.178631, 'y': 0.251748, 'z': -0.046175},
        'earbuds': {'x': 0.624219, 'y': 2.335583, 'z': -0.595254},
        'left_watch': {'x': 0.099619, 'y': 0.138018, 'z': 0.024356},
        'right_watch': {'x': -0.265199, 'y': -0.442142, 'z': 0.087390}
    }
    
    corrected_dfs = []
    
    # Process each dataframe
    for i, df in enumerate(aligned_dfs):
        if df is None:
            corrected_dfs.append(None)
            continue
            
        device_name = df_names[i]
        if device_name not in device_biases:
            corrected_dfs.append(df)  # Keep unchanged if no bias values
            continue
            
        corrected_df = df.copy()
        
        # Check if gyroscope columns exist in the dataframe
        gyro_axes = ['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
        has_gyro_columns = all(col in corrected_df.columns for col in gyro_axes)
        
        if has_gyro_columns:
            print(f"Applying bias correction to {device_name} gyroscope data:")
            
            # Print original means
            original_means = {axis: corrected_df[axis].mean() for axis in gyro_axes}
            print(f"  Original means: X={original_means['x-axis (deg/s)']:.6f}, "
                  f"Y={original_means['y-axis (deg/s)']:.6f}, "
                  f"Z={original_means['z-axis (deg/s)']:.6f}")
            
            # Apply bias correction using a mapping between axes and bias values
            axis_to_bias = {
                'x-axis (deg/s)': 'x',
                'y-axis (deg/s)': 'y',
                'z-axis (deg/s)': 'z'
            }
            
            for axis, bias_key in axis_to_bias.items():
                corrected_df[axis] = corrected_df[axis] - device_biases[device_name][bias_key]
            
            # Print corrected means
            corrected_means = {axis: corrected_df[axis].mean() for axis in gyro_axes}
            print(f"  Corrected means: X={corrected_means['x-axis (deg/s)']:.6f}, "
                  f"Y={corrected_means['y-axis (deg/s)']:.6f}, "
                  f"Z={corrected_means['z-axis (deg/s)']:.6f}")
        
        corrected_dfs.append(corrected_df)
    
    return corrected_dfs
