import pandas as pd
import numpy as np

"""WATCH SENSOR ALIGNER: First step in pipeline, taking raw wtach data and gathering the correct 
headers and timestamps.

1. Initialsie Column names as variables that might contain different types of data
2. Find Next Column: Helper function that takes a dataframe + a list of possible column names and returns the first column name that exists in the dataframe 
3. Extract Time: Extract specififc time period of a string by cutting off after 'T' character
4. Extract Ms From Time: Checks if the timestamp contains milliseconds and if so extracts as an integer 
5. Process Timestamp: Input dataframe and returns a modified version with standardised timestamps using extract time method
6. Align Sensor Data: Takes accelerometer and gyroscope dataframes and aligns them
7. Adjust Column Length:Truncating extra values if source df is longer than target/Padding with the last value if source is shorter than target
 """
class WatchSensorAligner:
    def __init__(self):
       #Initialise variables for diff column names
       self.timestamp_columns = ['timestamp (+0000)', 'timestamp', 'time (-00:00)']
       self.epoch_columns = ['epoc (ms)', 'epoch', 'epoch (ms)']
       self.data_collection_columns = {
           'accel': ['x-axis (g)', 'y-axis (g)', 'z-axis (g)'],
           'gyro': ['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
       }



    def find_next_column(self, df, possible_columns):
        for column in possible_columns:
            # If column exists retur first match
            if column in df.columns:
                return column
        return None



    def extract_time(self, timestamp):
        timestamp_string = str(timestamp)
    
        # Check if this is an ISO format timestamp with a 'T' separator
        if 'T' in timestamp_string:
            # Split by 'T' and take the second part (the time portion)
            time_part_cut = timestamp_string.split('T')[1]
            time_part_colons = time_part_cut.replace('.', ':')
            time_final = time_part_colons.split('+')[0]
            
            return time_final
        return timestamp_string



    def extract_ms_from_time(self, timestamp):
        timestamp_string = str(timestamp) #Timestamp to string
        if '.' in timestamp_string:
            milliseconds_part = timestamp_string.split('.')[-1] #split on miliseconds 
            return int(milliseconds_part) #convert to int 
        #otherwise return 0
        return 0



    def process_timestamp(self, df):
        timestamp_column = self.find_next_column(df, self.timestamp_columns)
        #Error message if empty timestamp column
        if not timestamp_column:
            print(f"No timestamp column found. Available columns: {df.columns.tolist()}")
            return None
        
        try: #applies the extract time method to standardise the format
            df['timestamp'] = df[timestamp_column].apply(self.extract_time)
            cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
            return df[cols]
        except Exception:
            print("Error processing timestamps")
            return None



    def align_sensor_data(self, accelerometer_df, gyroscope_df):
        accelerometer_df = accelerometer_df.copy()
        gyroscope_df = gyroscope_df.copy()
        # Ensure 'timestamp' column exists
        accelerometer_df = self.process_timestamp(accelerometer_df)
        gyroscope_df = self.process_timestamp(gyroscope_df)
        if accelerometer_df is None or gyroscope_df is None:
            return None
        
        # MS timestamps for speciffc  matching
        accelerometer_epochms_column = self.find_next_column(accelerometer_df, self.epoch_columns)
        gyroscope_epochms_column = self.find_next_column(gyroscope_df, self.epoch_columns)
        
        if accelerometer_epochms_column is not None and gyroscope_epochms_column is not None:
            # Use epoch milliseconds for prcise alignment 
            accelerometer_df['timestamp_ms'] = accelerometer_df[accelerometer_epochms_column]
            gyroscope_df['timestamp_ms'] = gyroscope_df[gyroscope_epochms_column]
        else:
            # Extract milliseconds from timestamp strings
            accelerometer_df['timestamp_ms'] = accelerometer_df['timestamp'].apply(self.extract_ms_from_time)
            gyroscope_df['timestamp_ms'] = gyroscope_df['timestamp'].apply(self.extract_ms_from_time)
        
        # Create output dataframe
        merged_df = pd.DataFrame()
        merged_df['timestamp'] = accelerometer_df['timestamp']
        
        # Add sensor columns with fallback to NaN
        for sensor_type, column in self.data_collection_columns.items():
            source_df = accelerometer_df if sensor_type == 'accel' else gyroscope_df
            
            for row in column:
                if row in source_df.columns:
                    if sensor_type == 'accel':
                        merged_df[row] = source_df[row]
                    else:
                        # Handle gyroscope data length differences
                        self.adjust_column_lengh(merged_df, source_df, row)
                else:
                    merged_df[row] = np.nan   # Add NaN values for missing columns
        
        return merged_df
            

    
    def adjust_column_lengh(self, target_df, source_df, column_name):
        values_to_add = source_df[column_name].values  # values from the df column
        
        # Check if length adjustment is needed
        target_length = len(target_df)
        source_length = len(values_to_add)
        
        if source_length != target_length:
            if source_length > target_length: #truncate
                adjusted_values = values_to_add[:target_length]
            else: #add padding
                padding_length = target_length - source_length
                last_value = values_to_add[-1]
                padding_values = np.full(padding_length, last_value)
                adjusted_values = np.concatenate([values_to_add, padding_values])
        else:
            adjusted_values = values_to_add
        
        # Add the adjusted column to the target DataFrame
        target_df[column_name] = adjusted_values


def correct_gyroscope_bias(df_to_calibrate):
    df_names = ['phone', 'earbuds', 'left_watch', 'right_watch']
    
    # Hardcode the calibration bias
    calibration_bias = {
        'phone': {'x': 0.178631, 'y': 0.251748, 'z': -0.046175},
        'earbuds': {'x': 0.624219, 'y': 2.335583, 'z': -0.595254},
        'left_watch': {'x': 0.099619, 'y': 0.138018, 'z': 0.024356},
        'right_watch': {'x': -0.265199, 'y': -0.442142, 'z': 0.087390}
    }
    #Maps gyroscope column names to bias keys
    gyroscope_axis = ['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
    axis_dict = {
        'x-axis (deg/s)': 'x',
        'y-axis (deg/s)': 'y',
        'z-axis (deg/s)': 'z'
    }

    calibrated_df = []    #List for calirated dfs 
    for i, df in enumerate(df_to_calibrate):
        if df is None:
            calibrated_df.append(None)
            continue   
        device_name = df_names[i]
        corrected_df = df.copy()

        # Apply bias correction by subtracting the device-specific bias values
        for axis, bias_value in axis_dict.items(): 
            corrected_df[axis] = corrected_df[axis] - calibration_bias[device_name][bias_value]

        #Out put corrected calibrated values 
        corrected_means = {axis: corrected_df[axis].mean() for axis in gyroscope_axis}
        print(f"  Corrected means: X={corrected_means['x-axis (deg/s)']:.6f}, "
                f"Y={corrected_means['y-axis (deg/s)']:.6f}, "
                f"Z={corrected_means['z-axis (deg/s)']:.6f}")
    
    calibrated_df.append(corrected_df)
    
    return calibrated_df