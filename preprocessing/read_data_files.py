import pandas as pd 
import os 
import glob

def read_csv(file_path, nrows=None):
    """Read CSV with robust parameter settings to handle encoding and whitespace issues"""
    try:
        return pd.read_csv(
            file_path, 
            nrows=nrows,
            skipinitialspace=True,  # Whitespace after commas
            encoding='utf-8',       # Set encoding
            dtype=str if nrows == 1 else None  # For headers, read as strings
        )
    except Exception as e:
        print(f"Error reading {file_path} with enhanced parameters: {e}")

def is_accelerometer_file(file_path):
    """Check if a file contains accelerometer data by checking its filename"""
    try:
        if "accelerometer" in os.path.basename(file_path).lower():
            return True
    except Exception as e:
        print(f"  Error checking file {file_path}: {e}")
        return False

def is_gyroscope_file(file_path):
    """Check if a file contains gyroscope data by checking its filename"""
    try:
        if "gyroscope" in os.path.basename(file_path).lower():
            return True
    except Exception as e:
        print(f"  Error checking file {file_path}: {e}")
        return False
    
#Determine the type of watch file and assign it to the appropriate category.
def assign_watch_file(file, data_files):
    basename = os.path.basename(file).lower()
    
    # Determine side (left/right)
    if 'left' in basename:
        side = 'left'
    elif 'right' in basename:
        side = 'right'
    else:
        return False  # Not a watch file
    
    # Determine sensor type (accel/gyro)
    is_accel_by_name = 'accelerometer' in basename
    is_gyro_by_name = 'gyroscope' in basename
    
    # Form the key for the data_files dictionary
    if is_accel_by_name:
        key = f'{side}_accel'
        sensor_type = 'accelerometer'
    elif is_gyro_by_name:
        key = f'{side}_gyro'
        sensor_type = 'gyroscope'
    # If not identifiable by name, check content
    elif is_accelerometer_file(file):
        key = f'{side}_accel'
        sensor_type = 'accelerometer (by content)'
    elif is_gyroscope_file(file):
        key = f'{side}_gyro'
        sensor_type = 'gyroscope (by content)'
    else:
        return False  # Cannot determine sensor type
    
    # If slot is already filled, skip
    if data_files[key] is not None:
        return False
    
    # Assign the file
    data_files[key] = file
    print(f"Found {side} watch {sensor_type} file: {os.path.basename(file)}")
    return True


def find_sensor_files(data_directory):
    # Initialize result dictionary with None values
    data_files = {
        'phone': None,
        'earbud': None,
        'left_accel': None,
        'left_gyro': None,
        'right_accel': None,
        'right_gyro': None
    }
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {data_directory}")
        return data_files
    
    # Create a copy we can modify
    remaining_files = list(csv_files)
    
    # 1. First identify earbuds file
    for file in list(remaining_files):
        basename = os.path.basename(file).lower()
        if 'esense' in basename:
            data_files['earbud'] = file
            remaining_files.remove(file)
            print(f"Found earbud file: {os.path.basename(file)}")
            break
    
    # 2. Process all watch files in a single pass
    for file in list(remaining_files):
        if assign_watch_file(file, data_files):
            remaining_files.remove(file)
    
    # 3. Handle phone file (remaining files)
    if remaining_files and data_files['phone'] is None:
        if len(remaining_files) > 1:
            # Find the largest file as it's most likely to be the phone data
            file_sizes = {f: os.path.getsize(f) for f in remaining_files}
            phone_file = max(file_sizes, key=file_sizes.get)
            print(f"Multiple potential phone files found. Using largest: {os.path.basename(phone_file)}")
        else:
            phone_file = remaining_files[0]
            print(f"Found phone file: {os.path.basename(phone_file)}")
        
        data_files['phone'] = phone_file
    
    # Print summary of found files
    print("\nSensor file detection summary:")
    for sensor_type, file_path in data_files.items():
        status = "✓" if file_path else "✗"
        print(f"  {sensor_type}: {status}")
    
    return data_files
