import pandas as pd 
import os 
import glob
"""READ DATA FILES: Collection of helper functions used for file reading and classification

1. Read CSV: Read CSV input with parameters to handle handle whitespace after comma and 
 to set character encoding to ensure any file name can be read 
2. Is Accelerometer File: Checks if a file contains accelerometer data by checking its filename 
3. Is Gyroscope File: Checks if a file contains gyroscope data by checking its filename
4. Assign Watch File: Determine the type of watch file by checking file names and sensor types.
After forms a key for the data dictionary and assigns file to category if empty (true is sucessful).
5. Find Sensor File: Initialises dictionary for different sensor types, finds all CSV files in the 
and processes files """

def read_csv(file_path ):
    try:
        return pd.read_csv(
            file_path, 
            nrows=None,
            skipinitialspace=True,  # Whitespace after commas
            encoding='utf-8',       # Set encoding
        )
    except Exception:
        print(f"Error reading {file_path}")

def is_accelerometer_file(file_path):
    try:
        if "accelerometer" in os.path.basename(file_path).lower():
            return True
    except Exception:
        print(f"Error checking file {file_path}")
        return False

def is_gyroscope_file(file_path):
    try:
        if "gyroscope" in os.path.basename(file_path).lower():
            return True
    except Exception:
        print(f"Error checking file {file_path}")
        return False
    

def assign_watch_file(file, data_dict):
    filename = os.path.basename(file).lower()
    #Left or right device
    if 'left' in filename:
        side = 'left'
    elif 'right' in filename:
        side = 'right'
    else:
        return False  # Not a watch file
    
    # Gyro or accel sensor type
    is_accel_by_name = 'accelerometer' in filename
    is_gyro_by_name = 'gyroscope' in filename
    
    # Key for the data_files dict
    if is_accel_by_name:
        dict_key = f'{side}_accel'
        sensor_type = 'accelerometer'
    elif is_gyro_by_name:
        dict_key = f'{side}_gyro'
        sensor_type = 'gyroscope'
    #Otherwise check content
    elif is_accelerometer_file(file):
        dict_key = f'{side}_accel'
        sensor_type = 'accelerometer (by content)'
    elif is_gyroscope_file(file):
        dict_key = f'{side}_gyro'
        sensor_type = 'gyroscope (by content)'
    else:
        return False
    
    # If slot is already filled, skip
    if data_dict[dict_key] is not None:
        return False
    
    # If slot is free ssign the file
    data_dict[dict_key] = file
    print(f"Found {side} watch {sensor_type} file: {os.path.basename(file)}")
    return True


def find_sensor_files(data_dict):
    data_files = { #initialise data dictionary
        'phone': None,
        'earbud': None,
        'left_accel': None,
        'left_gyro': None,
        'right_accel': None,
        'right_gyro': None
    }
    
    # retrrieve all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dict, "*.csv"))
    
    if not csv_files:
        print("No CSV files found")
        return data_files
    
    remaining_files = list(csv_files)
    
    # 1. earbuds file
    for file in list(remaining_files):
        filename = os.path.basename(file).lower()
        if 'esense' in filename:
            data_files['earbud'] = file # type: ignore
            remaining_files.remove(file)
            print(f"Found earbud file: {os.path.basename(file)}")
            break #max 1 earbud 
    
    # 2. Process watch files
    for file in list(remaining_files):
        if assign_watch_file(file, data_files):
            remaining_files.remove(file)
    
    # 3. Remaining file is phone file
    if remaining_files and data_files['phone'] is None:
        if len(remaining_files) > 1:
            # Find the largest file as it's most likely to be the phone data
            file_sizes = {f: os.path.getsize(f) for f in remaining_files}
            phone_file = max(file_sizes, key=file_sizes.get) # type: ignore
            print(f"Multiple potential phone files found. Using largest: {os.path.basename(phone_file)}")
        else:
            phone_file = remaining_files[0]
            print(f"Found phone file: {os.path.basename(phone_file)}")
        
        data_files['phone'] = phone_file # type: ignore
    
    # Print summary of found files
    print("\nSensor file detection summary:")
    for sensor_type, file_path in data_files.items():
        status = "Availible" if file_path else "Unavailible"
        print(f"  {sensor_type}: {status}")
    
    return data_files
