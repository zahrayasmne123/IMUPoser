from .phone_processor import PhoneSensorAligner
from .watch_processor import WatchSensorAligner
from .earbuds_processor import EarbudSensorAligner
from .rotation_processor import robust_rotation_matrices_dataframes
from .synchronise_dataframes import robust_synchronise_dataframes
from .csvtotensor import create_IMUPoser_tensor
from .trim_timestamps import trim_dataframes
import pandas as pd
import glob
import os

def read_csv_safely(file_path, nrows=None):
    """Read CSV with robust parameter settings to handle encoding and whitespace issues"""
    try:
        return pd.read_csv(
            file_path, 
            nrows=nrows,
            skipinitialspace=True,  # Handle whitespace after commas
            encoding='utf-8',       # Explicitly set encoding
            quotechar='"',          # Explicitly set quote character
            dtype=str if nrows == 1 else None  # For headers, read as strings
        )
    except Exception as e:
        print(f"Error reading {file_path} with enhanced parameters: {e}")
        # Fallback to minimal parameters
        try:
            return pd.read_csv(file_path, nrows=nrows)
        except Exception as e2:
            print(f"Error with fallback method: {e2}")
            raise

def is_accelerometer_file(file_path):
    """Check if a file contains accelerometer data by checking its filename"""
    try:
        # IMPORTANT: Trust the filename over the contents
        # If it has "Accelerometer" in the name, it IS an accelerometer file
        if "accelerometer" in os.path.basename(file_path).lower():
            print("Identified as accelerometer file based on filename")
            return True
        
        # If no clue from filename, check the headers as fallback
        print(f"Checking if {file_path} is an accelerometer file")
        # Use our improved CSV reader
        df = read_csv_safely(file_path, nrows=1)
        cols = df.columns.tolist()
        print(f"  Found columns: {cols}")
        
        # Check if the file has x-axis, y-axis, z-axis columns with (g) unit
        x_accel = any('x-axis' in col.lower() and '(g)' in col for col in cols)
        y_accel = any('y-axis' in col.lower() and '(g)' in col for col in cols)
        z_accel = any('z-axis' in col.lower() and '(g)' in col for col in cols)
        has_accel = x_accel and y_accel and z_accel
        print(f"  Contains accelerometer data: {has_accel}")
        return has_accel
    except Exception as e:
        print(f"  Error checking file {file_path}: {e}")
        return False

def is_gyroscope_file(file_path):
    """Check if a file contains gyroscope data by checking its filename"""
    try:
        # IMPORTANT: Trust the filename over the contents
        # If it has "Gyroscope" in the name, it IS a gyroscope file regardless of column headers
        if "gyroscope" in os.path.basename(file_path).lower():
            print("Identified as gyroscope file based on filename")
            return True
            
        # If no clue from filename, check the headers as fallback
        print(f"Checking if {file_path} is a gyroscope file")
        # Use our improved CSV reader
        df = read_csv_safely(file_path, nrows=1)
        cols = df.columns.tolist()
        print(f"  Found columns: {cols}")
        
        # Check if the file has x-axis, y-axis, z-axis columns with (deg/s) unit
        x_gyro = any('x-axis' in col.lower() and '(deg/s)' in col for col in cols)
        y_gyro = any('y-axis' in col.lower() and '(deg/s)' in col for col in cols)
        z_gyro = any('z-axis' in col.lower() and '(deg/s)' in col for col in cols)
        has_gyro = x_gyro and y_gyro and z_gyro
        print(f"  Contains gyroscope data: {has_gyro}")
        return has_gyro
    except Exception as e:
        print(f"  Error checking file {file_path}: {e}")
        return False

def fix_gyroscope_columns(gyro_df):
    """
    Fix gyroscope dataframe that has incorrect column headers (g instead of deg/s)
    This can happen if gyroscope files incorrectly use accelerometer column labels
    """
    if gyro_df is None:
        return None
        
    # Check if we need to fix the columns
    cols = gyro_df.columns.tolist()
    has_g_units = any('(g)' in col for col in cols)
    has_degs_units = any('(deg/s)' in col for col in cols)
    
    # If it has (g) units but not (deg/s) units, it needs fixing
    if has_g_units and not has_degs_units:
        print("CRITICAL FIX: Correcting gyroscope column headers that incorrectly use (g) units")
        print("Original columns:", cols)
        
        renamed_cols = {}
        for col in cols:
            if '(g)' in col:
                renamed_cols[col] = col.replace('(g)', '(deg/s)')
                
        # Rename the columns
        gyro_df = gyro_df.rename(columns=renamed_cols)
        print(f"Fixed columns: {gyro_df.columns.tolist()}")
        
    return gyro_df

def find_sensor_files(data_directory):
    """
    Dynamically detect sensor files in a directory using pattern matching.
    
    Args:
        data_directory: Directory containing the sensor data CSV files
        
    Returns:
        Dictionary of detected file paths for each sensor type
    """
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
    
    # Detect earbuds file - contains 'esense' in filename
    earbud_files = [f for f in csv_files if 'esense' in f.lower()]
    if earbud_files:
        data_files['earbud'] = earbud_files[0] # type: ignore
        print(f"Found earbud file: {os.path.basename(data_files['earbud'])}")
        # Remove from list to avoid double-matching
        csv_files = [f for f in csv_files if f not in earbud_files]
    
    # First pass - identify files by left/right and accelerometer/gyroscope in filename
    for file in csv_files[:]:
        basename = os.path.basename(file).lower()
        
        # Identify left/right watch files by filename first
        is_left = 'left' in basename
        is_right = 'right' in basename
        
        # Then check for accelerometer/gyroscope files by filename
        is_accel_by_name = 'accelerometer' in basename
        is_gyro_by_name = 'gyroscope' in basename
        
        # Assign based on filename patterns with high confidence
        if is_left and is_accel_by_name and data_files['left_accel'] is None:
            data_files['left_accel'] = file  # type: ignore
            print(f"Found left watch accelerometer file (by filename): {os.path.basename(file)}")
            csv_files.remove(file)
        elif is_left and is_gyro_by_name and data_files['left_gyro'] is None:
            data_files['left_gyro'] = file  # type: ignore
            print(f"Found left watch gyroscope file (by filename): {os.path.basename(file)}")
            csv_files.remove(file)
        elif is_right and is_accel_by_name and data_files['right_accel'] is None:
            data_files['right_accel'] = file  # type: ignore
            print(f"Found right watch accelerometer file (by filename): {os.path.basename(file)}")
            csv_files.remove(file)
        elif is_right and is_gyro_by_name and data_files['right_gyro'] is None:
            data_files['right_gyro'] = file  # type: ignore
            print(f"Found right watch gyroscope file (by filename): {os.path.basename(file)}")
            csv_files.remove(file)
    
    # Remaining files - identify by content if needed
    # First separate left/right
    left_files = [f for f in csv_files if 'left' in os.path.basename(f).lower()]
    right_files = [f for f in csv_files if 'right' in os.path.basename(f).lower()]
    
    # Remove identified files to avoid double-matching
    other_files = [f for f in csv_files if f not in left_files and f not in right_files]
    
    # Process left watch files
    if left_files and (data_files['left_accel'] is None or data_files['left_gyro'] is None):
        print(f"\nProcessing {len(left_files)} remaining left watch files...")
        for file in left_files:
            if is_accelerometer_file(file) and data_files['left_accel'] is None:
                data_files['left_accel'] = file  # type: ignore
                print(f"Found left watch accelerometer file: {os.path.basename(file)}")
            if is_gyroscope_file(file) and data_files['left_gyro'] is None:
                data_files['left_gyro'] = file  # type: ignore
                print(f"Found left watch gyroscope file: {os.path.basename(file)}")
    
    # Process right watch files
    if right_files and (data_files['right_accel'] is None or data_files['right_gyro'] is None):
        print(f"\nProcessing {len(right_files)} remaining right watch files...")
        for file in right_files:
            if is_accelerometer_file(file) and data_files['right_accel'] is None:
                data_files['right_accel'] = file  # type: ignore
                print(f"Found right watch accelerometer file: {os.path.basename(file)}")
            if is_gyroscope_file(file) and data_files['right_gyro'] is None:
                data_files['right_gyro'] = file  # type: ignore
                print(f"Found right watch gyroscope file: {os.path.basename(file)}")
    
    # Handle phone files
    # Collect list of all files already assigned to a sensor
    assigned_files = [f for f in [
        data_files['left_accel'], data_files['left_gyro'],
        data_files['right_accel'], data_files['right_gyro'],
        data_files['earbud']
    ] if f is not None]
    
    # Make sure we only consider truly unassigned files for phone
    unassigned_files = [f for f in other_files if f not in assigned_files]
    
    if unassigned_files:
        if len(unassigned_files) > 1:
            # Get file sizes
            file_sizes = {f: os.path.getsize(f) for f in unassigned_files}
            # Find the largest file
            phone_file = max(file_sizes, key=lambda f: file_sizes[f])
            print(f"Multiple potential phone files found. Using largest: {os.path.basename(phone_file)}")
        else:
            phone_file = unassigned_files[0]
            print(f"Found phone file: {os.path.basename(phone_file)}")
        
        data_files['phone'] = phone_file  # type: ignore
    else:
        print("No unassigned files available for phone data")
    
    # Print summary of found files
    print("\nSensor file detection summary:")
    for sensor_type, file_path in data_files.items():
        if file_path:
            print(f"  {sensor_type}: ✓")
        else:
            print(f"  {sensor_type}: ✗")
    
    return data_files

def align_all_sensor_data(data_directory):
    """
    Dynamically align all available sensor data from the given directory.
    Uses pattern matching to find the appropriate files.
    """
    # Initialize sensor aligners
    phone = PhoneSensorAligner()
    watch_aligner = WatchSensorAligner()
    earbuds_aligner = EarbudSensorAligner()

    # Find sensor files using pattern matching
    data_files = find_sensor_files(data_directory)

    # Initialize aligned dataframes to None
    phone_aligned_df = None
    earbud_aligned_df = None
    left_watch_aligned_df = None
    right_watch_aligned_df = None

    # Process phone data if available
    try:
        if data_files['phone']:
            # Use safe reading function
            phone_df = read_csv_safely(data_files['phone'])
            # Remove any unnamed columns
            phone_df = phone_df.drop(columns=[col for col in phone_df.columns if 'Unnamed:' in col], errors='ignore')
            phone_aligned_df = phone.align_sensor_data(phone_df)
            print("Successfully processed phone data")
        else:
            print("No phone data file found")
    except Exception as e:
        print(f"Error processing phone data: {str(e)}")

    # Process earbud data if available
    try:
        if data_files['earbud']:
            # Use safe reading function
            earbud_df = read_csv_safely(data_files['earbud'])
            earbud_aligned_df = earbuds_aligner.align_sensor_data(earbud_df)
            print("Successfully processed earbud data")
        else:
            print("No earbud data file found")
    except Exception as e:
        print(f"Error processing earbud data: {str(e)}")

    # Process left watch data if available
    try:
        if data_files['left_accel'] and data_files['left_gyro']:
            # Use safe reading function for both files
            leftaccel_df = read_csv_safely(data_files['left_accel'])
            leftgyro_df = read_csv_safely(data_files['left_gyro'])
            
            # Fix gyroscope column headers if needed
            leftgyro_df = fix_gyroscope_columns(leftgyro_df)
            
            # Debug info
            print(f"\nLeft accelerometer columns: {leftaccel_df.columns.tolist()}")
            print(f"Left gyroscope columns: {leftgyro_df.columns.tolist()}")  # type: ignore
            print(f"Left accelerometer data sample:\n{leftaccel_df.head(2)}")
            print(f"Left gyroscope data sample:\n{leftgyro_df.head(2)}")  # type: ignore

            left_watch_aligned_df = watch_aligner.align_sensor_data(leftaccel_df, leftgyro_df)
            print("Successfully processed left watch data")
        else:
            missing = []
            if not data_files['left_accel']:
                missing.append("accelerometer")
            if not data_files['left_gyro']:
                missing.append("gyroscope")
            print(f"Incomplete left watch data: missing {', '.join(missing)} file(s)")
    except Exception as e:
        print(f"Error processing left watch data: {str(e)}")

    # Process right watch data if available
    try:
        if data_files['right_accel'] and data_files['right_gyro']:
            # Use safe reading function for both files
            rightaccel_df = read_csv_safely(data_files['right_accel'])
            rightgyro_df = read_csv_safely(data_files['right_gyro'])
            
            # Fix gyroscope column headers if needed
            rightgyro_df = fix_gyroscope_columns(rightgyro_df)
            
            # Debug info
            print(f"\nRight accelerometer columns: {rightaccel_df.columns.tolist()}")
            print(f"Right gyroscope columns: {rightgyro_df.columns.tolist()}")  # type: ignore
            
            right_watch_aligned_df = watch_aligner.align_sensor_data(rightaccel_df, rightgyro_df)
            print("Successfully processed right watch data")
        else:
            missing = []
            if not data_files['right_accel']:
                missing.append("accelerometer")
            if not data_files['right_gyro']:
                missing.append("gyroscope")
            print(f"Incomplete right watch data: missing {', '.join(missing)} file(s)")
    except Exception as e:
        print(f"Error processing right watch data: {str(e)}")

    # Return all dataframes, even if some are None
    return phone_aligned_df, earbud_aligned_df, left_watch_aligned_df, right_watch_aligned_df

#Bias correction for gyroscope data based on calibration values 
def correct_gyroscope_bias(aligned_dfs, df_names=None):
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
        has_gyro_columns = any('(deg/s)' in col for col in corrected_df.columns)
        
        if has_gyro_columns:
            print(f"Applying bias correction to {device_name} gyroscope data:")
            print(f"  Original means: X={corrected_df['x-axis (deg/s)'].mean():.6f}, "
                  f"Y={corrected_df['y-axis (deg/s)'].mean():.6f}, "
                  f"Z={corrected_df['z-axis (deg/s)'].mean():.6f}")
            
            # Apply bias correction
            corrected_df['x-axis (deg/s)'] = corrected_df['x-axis (deg/s)'] - device_biases[device_name]['x']
            corrected_df['y-axis (deg/s)'] = corrected_df['y-axis (deg/s)'] - device_biases[device_name]['y']
            corrected_df['z-axis (deg/s)'] = corrected_df['z-axis (deg/s)'] - device_biases[device_name]['z']
            
            print(f"  Corrected means: X={corrected_df['x-axis (deg/s)'].mean():.6f}, "
                  f"Y={corrected_df['y-axis (deg/s)'].mean():.6f}, "
                  f"Z={corrected_df['z-axis (deg/s)'].mean():.6f}")
        
        corrected_dfs.append(corrected_df)
    
    return corrected_dfs


def process_aligned_sensor_data(aligned_dfs, df_names=None, output_path=None):
    """
    Process any available aligned sensor data and create an IMUPoser tensor.
    Handles missing dataframes gracefully.
    
    Args:
        aligned_dfs: List of aligned dataframes (can contain None values)
        df_names: Names corresponding to each dataframe
        output_path: Path to save the output tensor (optional)
        
    Returns:
        Tuple of (synchronized dataframes, IMUPoser tensor)
    """
    if df_names is None:
        df_names = ['phone', 'earbuds', 'left_watch', 'right_watch']

    # Validate inputs
    if len(aligned_dfs) != len(df_names):
        raise ValueError(f"Number of dataframes ({len(aligned_dfs)}) must match number of names ({len(df_names)})")

    # Filter out None values
    valid_dfs = []
    valid_names = []
    for i, df in enumerate(aligned_dfs):
        if df is not None:
            valid_dfs.append(df)
            valid_names.append(df_names[i])
            print(f"Valid dataframe: {df_names[i]} with {len(df)} rows")
        else:
            print(f"Skipping None dataframe: {df_names[i]}")

    if not valid_dfs:
        raise ValueError("No valid dataframes to process")

    # Step 1: Trim dataframes
    print("\nTrimming dataframes...")
    trimmed_dfs_list = trim_dataframes(valid_dfs, valid_names)

    # Step 2: Calculate rotation matrices
    print("\nCalculating rotation matrices...")
    rotated_trimmed_dfs_list = robust_rotation_matrices_dataframes(trimmed_dfs_list)

    # Step 3: Synchronize dataframes
    print("\nSynchronizing dataframes...")
    synced_dfs = robust_synchronise_dataframes(rotated_trimmed_dfs_list, valid_names)

    # Step 4: Create IMUPoser tensor
    print("\nCreating IMUPoser tensor...")
    
    # Map the synchronized dataframes back to their original positions
    # We need to create a list with Nones for missing devices
    full_synced_dfs = [None] * len(df_names)
    for i, name in enumerate(valid_names):
        original_index = df_names.index(name)
        full_synced_dfs[original_index] = synced_dfs[i]

    
    # Create tensor with proper device names
    tensor = create_IMUPoser_tensor(full_synced_dfs, device_names=df_names, output_path=output_path) # type: ignore

    return synced_dfs, tensor


def full_sensor_pipeline(data_dir='rawdata', output_path=None):
    """
    Complete pipeline that dynamically processes any available sensor data.
    
    Args:
        data_dir: Directory containing the sensor data CSV files
        output_path: Path to save the output tensor (optional)
        
    Returns:
        Tuple of (synchronized dataframes, IMUPoser tensor)
    """
    # Step 1: Align any available sensor data
    print("Aligning sensor data...")
    aligned_dfs = align_all_sensor_data(data_dir)

    # Step 2: Process aligned data
    # Use standardized device names that match IMUPoser's expected mapping
    df_names = ['phone', 'earbuds', 'left_watch', 'right_watch']
    synced_dfs, tensor = process_aligned_sensor_data(aligned_dfs, df_names, output_path)

    print("\nPipeline completed successfully!")
    print(f"Number of active devices: {sum(1 for df in aligned_dfs if df is not None)}")
    
    return synced_dfs, tensor

