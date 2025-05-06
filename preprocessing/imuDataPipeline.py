from .phone_processor import PhoneSensorAligner
from .watch_processor import WatchSensorAligner
from .earbuds_processor import EarbudSensorAligner
from .rotation_processor import rotation_matrix_df_conversion
from .synchronise_dataframes import sync_dataframes
from .csvtotensor import create_IMUPoser_tensor
from .trim_timestamps import trim_dataframes
from .read_data_files import read_csv, find_sensor_files

""" IMU DATA PIPELINE SUMMARY: The full sensor data processing pipeline for wearable devices. The pipeline takes raw sensor data from
IMUs, aligns timestamps, processes according to device requirements, trims to compatible time ranges, calculates rotation
matrices for gyroscopes data, synchronisies the data from all devices to ensure measurements are aligned in time,
and finally converts into a tensor format compatible with the IMUPoser model

1.Process Single Data: takes single data file, aligner class, sensor name, and a file reading function
  Tries to read the data file using safe read_csv function and if sucessful applies the correct data aligner
2.Process Two Data Files: Separate processor that takes two input files used specifically for watch devices which produce separate files for
  different sensor types 
3.Process Aligned Sensor Data: Takes a list of dfs with sensor data from each device
   Ensures the number of dfs matches the number of device names and filters empty dfs
   Steps through preprocessing pipeline: trim, rotation matrix, data synchronisation, tensor creation
4. Full Sensor Pipeline: Aligns device data and feeds through pipelines using previously defined functions """


def process_single_data_file(data_file, device_aligner, sensor_name):
    try:
        if data_file:
            df = read_csv(data_file) #reads data file
            aligned_df = device_aligner.align_sensor_data(df) #applies device aligner 
            print(f"Successfully processed {sensor_name} data")
            return aligned_df
        # Error messages if device file isnt found 
        else:
            print(f"No {sensor_name} data file found")
            return None
    except Exception as e:
        print(f"Error processing {sensor_name} data: {str(e)}")
        return None



def process_two_data_files(accel_file, gyro_file, watch_aligner, left_or_right_side):
    try:
        if accel_file and gyro_file: #reads both accelerometer and gyroscope data files
            accel_df = read_csv(accel_file)
            gyro_df = read_csv(gyro_file)
            
            #calls the watch aligner to combine and align the accelerometer and gyroscope data
            aligned_df = watch_aligner.align_sensor_data(accel_df, gyro_df)
            print(f"Successfully processed {left_or_right_side} watch data")
            return aligned_df
        
        #error handling with specific messages about which files are missing 
        else:
            missing = []
            if not accel_file:
                missing.append("accelerometer")
            if not gyro_file:
                missing.append("gyroscope")
            print(f"Incomplete {left_or_right_side} watch data: missing {', '.join(missing)} file(s)")
            return None
    except Exception as e:
        print(f"Error processing {left_or_right_side} watch data: {str(e)}")
        return None




def align_all_sensor_data(data_directory):
    phone_aligner = PhoneSensorAligner()
    watch_aligner = WatchSensorAligner()
    earbuds_aligner = EarbudSensorAligner()

    # Find sensor files using pattern matching
    data_files = find_sensor_files(data_directory)

    # Process each sensor type
    phone_aligned_df = process_single_data_file(
        data_files['phone'], phone_aligner, 'phone')
    
    earbud_aligned_df = process_single_data_file(
        data_files['earbud'], earbuds_aligner, 'earbud')
    
    left_watch_aligned_df = process_two_data_files(
        data_files['left_accel'], data_files['left_gyro'], 
        watch_aligner, 'left')
    
    right_watch_aligned_df = process_two_data_files(
        data_files['right_accel'], data_files['right_gyro'], 
        watch_aligner, 'right')

    # Return all dataframes, even if some are None
    return phone_aligned_df, earbud_aligned_df, left_watch_aligned_df, right_watch_aligned_df




def process_aligned_sensor_data(aligned_dfs, output_path=None):
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

    #1. Trim dataframes
    print("\nTrimming dataframes...")
    trimmed_dfs_list = trim_dataframes(valid_dfs, valid_names)

    #2. Calculate rotation matrices
    print("\nCalculating rotation matrices...")
    rotated_trimmed_dfs_list = rotation_matrix_df_conversion(trimmed_dfs_list)

    # 3. Synchronize dataframes
    print("\nSynchronizing dataframes...")
    synced_dfs = sync_dataframes(rotated_trimmed_dfs_list, valid_names)

    #4. Create IMUPoser tensor
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
    synced_dfs, tensor = process_aligned_sensor_data(aligned_dfs, output_path)

    print("\nPipeline completed successfully!")
    return synced_dfs, tensor

