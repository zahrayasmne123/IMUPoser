import numpy as np
import torch

"""CSV TO TENSOR: Final step in preprocessing pipeline, converting the input accelerometer and gyroscope 
data to tensors to be put in the IMUPoser deep learning algorithm  


1.Validate_input_data: Input list of (dataframes) and corresponding device name to make a pair
    If there is a valid df, checks each dataframe to ensure all expected columns are present, if missing raise an error
    Check timestamp consistency and prints confirmation messages if validation passes
2.Df_to_numpy_array: Extracts accelerometer and rotation matrix data combining them into a single numpy array 
3.Create_IMUPoser_tensor: Builds tensor for input into IMUPoser model. Takes multiple device dataframes, validates the input data,
 and arranges the device data into specific positions. 

"""

def validate_input_data(dataframes, device_names):
    # Checks if there's at least one valid dataframe, error if noy
    df_device_names = [(df, name) for df, name in zip(dataframes, device_names) if df is not None]  
    valid_dataframes = [pair[0] for pair in df_device_names]
    valid_device_names = [pair[1] for pair in df_device_names]
    
    # Check for expected columns in each dataframe
    valid_column_names = ['timestamp'] + [f'{ax}-axis (m/s^2)' for ax in ['x', 'y', 'z']] + \
                   [f'R{i}{j}' for i in range(3) for j in range(3)]
    
    # Reports on missing columns 
    for df, name in zip(valid_dataframes, valid_device_names):
        missing_cols = set(valid_column_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} is missing columns: {missing_cols}")
    
    # Counts the number of rows in each dataframe checks all df have same number of rows
    row_counter = [len(df) for df in valid_dataframes]
    if len(set(row_counter)) != 1:
        raise ValueError("Inconsistent number of rows")
    
    # Check timestamp consistency
    base_timestamps = valid_dataframes[0]['timestamp'].values
    for df, name in zip(valid_dataframes[1:], valid_device_names[1:]):
        if not np.array_equal(base_timestamps, df['timestamp'].to_numpy()):
            raise ValueError(f"Timestamps don't match between {valid_device_names[0]} and {name}")
    return row_counter[0], valid_dataframes, valid_device_names


def df_to_numpy_array(df):
    accelerometer_data = df[[f'{ax}-axis (m/s^2)' for ax in ['x', 'y', 'z']]].values #Extract data
    gyroscope_data = df[[f'R{i}{j}' for i in range(3) for j in range(3)]].values #Extract data
    return np.concatenate([accelerometer_data, gyroscope_data], axis=1) #Concatenate into numpy array



def create_IMUPoser_tensor(dataframes_list, device_names,output_path):
    #Validate input data
    valid_names = [name for df, name in zip(dataframes_list, device_names) if df is not None]
    valid_dataframes_list = [df for df in dataframes_list if df is not None]
    n_frames, valid_dataframes, valid_device_names = validate_input_data(valid_dataframes_list, valid_names)
    
    # Define standard device mapping according to IMUPoser
    device_positions = {
        'phone': 0,        # Phone (left position)
        'left_watch': 1,   # Left watch
        'earbuds': 2,      # Earbuds/headphones
        'right_watch': 4   # Right watch
    }
    
    # Initialise tensor with zeros (n_frames x 60) and process each device in the correct position
    tensor_data = np.zeros((n_frames, 60))
    for df, name in zip(valid_dataframes, valid_device_names):
        position = None
        
        # Try to match the device name directly
        if name.lower() in device_positions:
            position = device_positions[name.lower()]
        if position is not None:
            device_data = df_to_numpy_array(df)
            tensor_data[:, position*12:(position+1)*12] = device_data
            print(f"✓ Placed device '{name}' at position {position}")
        else:
            print(f"⚠ Warning: No position mapping found for device '{name}', skipping")
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(tensor_data).float()
    if output_path:
        torch.save({'imu_data': tensor}, output_path)
        print(f"\nSaved tensor to {output_path}")
    return tensor