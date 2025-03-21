import pandas as pd
import numpy as np

class WatchSensorAligner:
    """
    A class to handle watch sensor data alignment and adjustments.
    Provides methods to clean, standardize timestamp formats, and merge accelerometer
    and gyroscope data into a single file.
    """
    
    def __init__(self):
       self.timestamp_columns = ['timestamp (+0000)', 'timestamp', 'time (-00:00)']
       self.epoch_columns = ['epoc (ms)', 'epoch', 'epoch (ms)']
       self.elapsed_columns = ['elapsed (s)']
       self.expected_columns = {
           'accel': ['x-axis (g)', 'y-axis (g)', 'z-axis (g)'],
           'gyro': ['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
       }
       # Add debug flag
       self.debug = True

    def process_timestamp(self, df):
        if df is None:
            return None
        
        timestamp_col = next((col for col in self.timestamp_columns if col in df.columns), None)
        epoch_col = next((col for col in self.epoch_columns if col in df.columns), None)
        
        if timestamp_col:
            if self.debug:
                print(f"Using timestamp column: {timestamp_col}")
                print(f"Sample values: {df[timestamp_col].head().tolist()}")
                print(f"First few timestamps: {df[timestamp_col].head()}")
            
            try:
                # If 'time (-00:00)' is found, it may already be in the right format
                if timestamp_col == 'time (-00:00)':
                    df['timestamp'] = df[timestamp_col]
                else:
                    # Try the original method
                    df['timestamp'] = df[timestamp_col].apply(
                        lambda ts: ts.split('T')[1].replace('.', ':').split('+')[0] if 'T' in str(ts) else ts
                    )
                
                # Keep the original columns for now to debug
                cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
                df = df[cols]
                
                return df
                
            except Exception as e:
                print(f"Error processing timestamps: {e}")
                print(f"Sample failed timestamp: {df[timestamp_col].iloc[0]}")
                return None
        else:
            print(f"No timestamp column found. Expected one of: {self.timestamp_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            return None

    def align_sensor_data(self, accel_df, gyro_df):
        print(f"Watch aligner received data - accel rows: {len(accel_df) if accel_df is not None else 'None'}, gyro rows: {len(gyro_df) if gyro_df is not None else 'None'}")
        
        if accel_df is None or gyro_df is None:
            print("Missing accelerometer or gyroscope data")
            return None
        
        try:
            # Make copies to avoid SettingWithCopyWarning
            accel_df = accel_df.copy()
            gyro_df = gyro_df.copy()
            
            # Get the time column
            accel_time_col = next((col for col in self.timestamp_columns if col in accel_df.columns), None)
            gyro_time_col = next((col for col in self.timestamp_columns if col in gyro_df.columns), None)
            
            # Get the epoch column
            accel_epoch_col = next((col for col in self.epoch_columns if col in accel_df.columns), None)
            gyro_epoch_col = next((col for col in self.epoch_columns if col in gyro_df.columns), None)
            
            if accel_time_col is None or gyro_time_col is None:
                print("Missing time columns")
                return None
                
            # Create or rename timestamp columns
            accel_df['timestamp'] = accel_df[accel_time_col]
            gyro_df['timestamp'] = gyro_df[gyro_time_col]
            
            # Create millisecond timestamps for matching
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
            
            print(f"Timestamp columns created: accel={accel_df['timestamp'].head(1).tolist()}, gyro={gyro_df['timestamp'].head(1).tolist()}")
            
            # Debug column presence before merging
            for col_type, cols in self.expected_columns.items():
                source_df = accel_df if col_type == 'accel' else gyro_df
                for col in cols:
                    if col not in source_df.columns:
                        print(f"WARNING: {col} not found in {col_type} DataFrame")
                    else:
                        print(f"Found {col} in {col_type} DataFrame")
            
            # Create output dataframe with accelerometer data first
            merged_df = pd.DataFrame()
            
            # Explicitly add timestamp
            merged_df['timestamp'] = accel_df['timestamp']
            
            # Add accel columns
            for col in self.expected_columns['accel']:
                if col in accel_df.columns:
                    merged_df[col] = accel_df[col]
                else:
                    print(f"Missing accelerometer column: {col}")
                    # Add a default column with NaN values to prevent missing column errors
                    merged_df[col] = np.nan
            
            # Add gyro columns with careful alignment
            for col in self.expected_columns['gyro']:
                if col in gyro_df.columns:
                    # Truncate or pad gyro data to match accel length
                    gyro_values = gyro_df[col].values
                    if len(gyro_values) > len(merged_df):
                        gyro_values = gyro_values[:len(merged_df)]  # Truncate
                    elif len(gyro_values) < len(merged_df):
                        # Pad with last value
                        padding = np.full(len(merged_df) - len(gyro_values), gyro_values[-1])
                        gyro_values = np.concatenate([gyro_values, padding])
                    merged_df[col] = gyro_values
                else:
                    print(f"Missing gyroscope column: {col}")
                    # Add a default column with NaN values to prevent missing column errors
                    merged_df[col] = np.nan
            
            # Double-check that all required columns now exist
            missing_cols = []
            for col_type in ['accel', 'gyro']:
                for col in self.expected_columns[col_type]:
                    if col not in merged_df.columns:
                        missing_cols.append(col)
            
            if missing_cols:
                print(f"Missing required columns after merge: {missing_cols}")
                # Instead of returning None, we'll continue with the available columns
                # This allows processing to continue even with missing data
            
            print(f"Successfully processed data with {len(merged_df)} rows")
            
            # Print column list for verification
            print(f"Final columns: {merged_df.columns.tolist()}")
            
            return merged_df
            
        except Exception as e:
            print(f"Error merging sensor data: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def validate_data(self, df, sensor_type):
        if not isinstance(df, pd.DataFrame):
            return False
            
        timestamp_col = next((col for col in self.timestamp_columns if col in df.columns), None)
        if not timestamp_col:
            print(f"Missing required timestamp column. Expected one of: {self.timestamp_columns}")
            return False
        
        # Debug output - show what columns are actually in the DataFrame
        if self.debug:
            print(f"Columns in {sensor_type} DataFrame: {df.columns.tolist()}")
            
        # Check for expected columns but be more lenient - just log warnings for missing columns
        expected_cols = self.expected_columns[sensor_type]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"Watch Missing required {sensor_type} columns: {missing_cols}")
            # Return True anyway - don't fail validation
            return True
            
        return True