from .phone_processor import PhoneSensorAligner
from .watch_processor import WatchSensorAligner
from .earbuds_processor import EarbudSensorAligner
from .rotation_processor import robust_rotation_matrices_dataframes
from .synchronise_dataframes import robust_synchronise_dataframes
from .csvtotensor import create_IMUPoser_tensor
from .trim_timestamps import trim_dataframes
from .read_data_files import read_csv, find_sensor_files



def process_sensor_data(data_file, processor, sensor_name, safe_reader=read_csv):
    """Process a single sensor data file and return the aligned dataframe."""
    try:
        if data_file:
            df = safe_reader(data_file)
            if sensor_name == 'phone':
                # Remove any unnamed columns for phone data
                df = df.drop(columns=[col for col in df.columns if 'Unnamed:' in col], errors='ignore')
            aligned_df = processor.align_sensor_data(df)
            print(f"Successfully processed {sensor_name} data")
            return aligned_df
        else:
            print(f"No {sensor_name} data file found")
            return None
    except Exception as e:
        print(f"Error processing {sensor_name} data: {str(e)}")
        return None



def process_watch_data(accel_file, gyro_file, watch_aligner, side, safe_reader=read_csv):
    """Process watch data with both accelerometer and gyroscope files."""
    try:
        if accel_file and gyro_file:
            accel_df = safe_reader(accel_file)
            gyro_df = safe_reader(gyro_file)
            
            # Debug info
            print(f"\n{side.capitalize()} accelerometer columns: {accel_df.columns.tolist()}")
            print(f"{side.capitalize()} gyroscope columns: {gyro_df.columns.tolist()}")
            
            # Only show sample data for left watch (to reduce redundancy)
            if side == 'left':
                print(f"{side.capitalize()} accelerometer data sample:\n{accel_df.head(2)}")
                print(f"{side.capitalize()} gyroscope data sample:\n{gyro_df.head(2)}")
                
            aligned_df = watch_aligner.align_sensor_data(accel_df, gyro_df)
            print(f"Successfully processed {side} watch data")
            return aligned_df
        else:
            missing = []
            if not accel_file:
                missing.append("accelerometer")
            if not gyro_file:
                missing.append("gyroscope")
            print(f"Incomplete {side} watch data: missing {', '.join(missing)} file(s)")
            return None
    except Exception as e:
        print(f"Error processing {side} watch data: {str(e)}")
        return None




def align_all_sensor_data(data_directory):
    phone_aligner = PhoneSensorAligner()
    watch_aligner = WatchSensorAligner()
    earbuds_aligner = EarbudSensorAligner()

    # Find sensor files using pattern matching
    data_files = find_sensor_files(data_directory)

    # Process each sensor type
    phone_aligned_df = process_sensor_data(
        data_files['phone'], phone_aligner, 'phone')
    
    earbud_aligned_df = process_sensor_data(
        data_files['earbud'], earbuds_aligner, 'earbud')
    
    left_watch_aligned_df = process_watch_data(
        data_files['left_accel'], data_files['left_gyro'], 
        watch_aligner, 'left')
    
    right_watch_aligned_df = process_watch_data(
        data_files['right_accel'], data_files['right_gyro'], 
        watch_aligner, 'right')

    # Return all dataframes, even if some are None
    return phone_aligned_df, earbud_aligned_df, left_watch_aligned_df, right_watch_aligned_df






def process_aligned_sensor_data(aligned_dfs, df_names=None, output_path=None):
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
    full_synced_dfs = [None] * len(df_names) # Create a list with Nones for missing devices
    for i, name in enumerate(valid_names):
        original_index = df_names.index(name)
        full_synced_dfs[original_index] = synced_dfs[i]

    tensor = create_IMUPoser_tensor(full_synced_dfs, device_names=df_names, output_path=output_path) # type: ignore

    return synced_dfs, tensor


def full_sensor_pipeline(data_dir='rawdata', output_path=None):
    print("Aligning sensor data...")
    aligned_dfs = align_all_sensor_data(data_dir)

    df_names = ['phone', 'earbuds', 'left_watch', 'right_watch']
    synced_dfs, tensor = process_aligned_sensor_data(aligned_dfs, df_names, output_path)

    print("\nPipeline completed successfully!")
    return synced_dfs, tensor

