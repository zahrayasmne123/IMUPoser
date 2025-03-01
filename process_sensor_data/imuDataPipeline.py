from phone_processor import PhoneSensorAligner
from watch_processor import WatchSensorAligner
from earbuds_processor import EarbudSensorAligner
from rotation_processor import robust_rotation_matrices_dataframes
from synchronise_dataframes import robust_synchronise_dataframes
from csvtotensor import create_IMUPoser_tensor
from trim_timestamps import trim_dataframes
import pandas as pd

def find_sensor_files(data_directory):
    """
    Dynamically detect sensor files in a directory using pattern matching.
    
    Args:
        data_directory: Directory containing the sensor data CSV files
        
    Returns:
        Dictionary of detected file paths for each sensor type
    """
    import os
    import glob
    
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
    
    # Detect left watch accelerometer - contains both 'left' and 'accelerometer'
    left_accel_files = [f for f in csv_files if 'left' in f.lower() and 'accelerometer' in f.lower()]
    if left_accel_files:
        data_files['left_accel'] = left_accel_files[0] # type: ignore
        print(f"Found left watch accelerometer file: {os.path.basename(data_files['left_accel'])}")
        # Remove from list
        csv_files = [f for f in csv_files if f not in left_accel_files]
    
    # Detect left watch gyroscope - contains both 'left' and 'gyroscope'
    left_gyro_files = [f for f in csv_files if 'left' in f.lower() and 'gyroscope' in f.lower()]
    if left_gyro_files:
        data_files['left_gyro'] = left_gyro_files[0] # type: ignore
        print(f"Found left watch gyroscope file: {os.path.basename(data_files['left_gyro'])}")
        # Remove from list
        csv_files = [f for f in csv_files if f not in left_gyro_files]
    
    # Detect right watch accelerometer - contains both 'right' and 'accelerometer'
    right_accel_files = [f for f in csv_files if 'right' in f.lower() and 'accelerometer' in f.lower()]
    if right_accel_files:
        data_files['right_accel'] = right_accel_files[0] # type: ignore
        print(f"Found right watch accelerometer file: {os.path.basename(data_files['right_accel'])}")
        # Remove from list
        csv_files = [f for f in csv_files if f not in right_accel_files]
    
    # Detect right watch gyroscope - contains both 'right' and 'gyroscope'
    right_gyro_files = [f for f in csv_files if 'right' in f.lower() and 'gyroscope' in f.lower()]
    if right_gyro_files:
        data_files['right_gyro'] = right_gyro_files[0] # type: ignore
        print(f"Found right watch gyroscope file: {os.path.basename(data_files['right_gyro'])}")
        # Remove from list
        csv_files = [f for f in csv_files if f not in right_gyro_files]
    
    # Any remaining file is likely phone data
    # If multiple files remain, take the largest one
    if csv_files:
        if len(csv_files) > 1:
            # Get file sizes
            file_sizes = {f: os.path.getsize(f) for f in csv_files}
            # Find the largest file
            phone_file = max(file_sizes, key=lambda f: file_sizes[f])
            print(f"Multiple potential phone files found. Using largest: {os.path.basename(phone_file)}")
        else:
            phone_file = csv_files[0]
            print(f"Found phone file: {os.path.basename(phone_file)}")
        
        data_files['phone'] = phone_file # type: ignore
    
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
            phone_df = pd.read_csv(data_files['phone'])
            phone_df = phone_df.drop(columns=[col for col in phone_df.columns if 'Unnamed:' in col])
            phone_aligned_df = phone.align_sensor_data(phone_df)
            print("Successfully processed phone data")
        else:
            print("No phone data file found")
    except Exception as e:
        print(f"Error processing phone data: {str(e)}")

    # Process earbud data if available
    try:
        if data_files['earbud']:
            earbud_df = pd.read_csv(data_files['earbud'])
            earbud_aligned_df = earbuds_aligner.align_sensor_data(earbud_df)
            print("Successfully processed earbud data")
        else:
            print("No earbud data file found")
    except Exception as e:
        print(f"Error processing earbud data: {str(e)}")

    # Process left watch data if available
    try:
        if data_files['left_accel'] and data_files['left_gyro']:
            leftaccel_df = pd.read_csv(data_files['left_accel'])
            leftgyro_df = pd.read_csv(data_files['left_gyro'])
            left_watch_aligned_df = watch_aligner.align_sensor_data(leftaccel_df, leftgyro_df)
            print("Successfully processed left watch data")
        else:
            if not data_files['left_accel']:
                print("No left watch accelerometer data file found")
            if not data_files['left_gyro']:
                print("No left watch gyroscope data file found")
    except Exception as e:
        print(f"Error processing left watch data: {str(e)}")

    # Process right watch data if available
    try:
        if data_files['right_accel'] and data_files['right_gyro']:
            rightaccel_df = pd.read_csv(data_files['right_accel'])
            rightgyro_df = pd.read_csv(data_files['right_gyro'])
            right_watch_aligned_df = watch_aligner.align_sensor_data(rightaccel_df, rightgyro_df)
            print("Successfully processed right watch data")
        else:
            if not data_files['right_accel']:
                print("No right watch accelerometer data file found")
            if not data_files['right_gyro']:
                print("No right watch gyroscope data file found")
    except Exception as e:
        print(f"Error processing right watch data: {str(e)}")

    # Return all dataframes, even if some are None
    return phone_aligned_df, earbud_aligned_df, left_watch_aligned_df, right_watch_aligned_df

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

