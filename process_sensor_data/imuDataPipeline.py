from .phone_processor import PhoneSensorAligner
from .watch_processor import WatchSensorAligner
from .earbuds_processor import EarbudSensorAligner
from .rotation_processor import robust_rotation_matrices_dataframes
from .synchronise_dataframes import robust_synchronise_dataframes
from .csvtotensor import create_IMUPoser_tensor
from .trim_timestamps import trim_dataframes

import os
import pandas as pd

def align_all_sensor_data(data_directory):
    # Initialize sensor aligners
    phone = PhoneSensorAligner()
    watch_aligner = WatchSensorAligner()
    earbuds_aligner = EarbudSensorAligner()

    # Load data files
    data_files = {
        'phone': os.path.join(data_directory, 'phonedata.csv'),
        'earbud': os.path.join(data_directory, 'earbuddata.csv'),
        'left_accel': os.path.join(data_directory, 'leftaccelerometerwatch.csv'),
        'left_gyro': os.path.join(data_directory, 'leftgyroscopewatch.csv'),
        'right_accel': os.path.join(data_directory, 'rightaccelerometerwatch.csv'),
        'right_gyro': os.path.join(data_directory, 'rightgyroscopewatch.csv')
    }

    # Read all CSV files
    try:
        phone_df = pd.read_csv(data_files['phone'])
        phone_df = phone_df.drop(columns=[col for col in phone_df.columns if 'Unnamed:' in col])
        earbud_df = pd.read_csv(data_files['earbud'])
        leftaccel_df = pd.read_csv(data_files['left_accel'])
        leftgyro_df = pd.read_csv(data_files['left_gyro'])
        rightaccel_df = pd.read_csv(data_files['right_accel'])
        rightgyro_df = pd.read_csv(data_files['right_gyro'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find required data file: {e.filename}")



    # Align sensor data
    phone_aligned_df = phone.align_sensor_data(phone_df)
    earbud_aligned_df = earbuds_aligner.align_sensor_data(earbud_df)
    left_watch_aligned_df = watch_aligner.align_sensor_data(leftaccel_df, leftgyro_df)
    right_watch_aligned_df = watch_aligner.align_sensor_data(rightaccel_df, rightgyro_df)

    return phone_aligned_df, earbud_aligned_df, left_watch_aligned_df, right_watch_aligned_df


def process_aligned_sensor_data(aligned_dfs, df_names=None):
    if df_names is None:
        df_names = ['Phone', 'Earbud', 'Left Watch', 'Right Watch']

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
    synced_dfs = robust_synchronise_dataframes(rotated_trimmed_dfs_list, 
                                     df_names)


    # Step 4: Create IMUPoser tensor
    print("\nCreating IMUPoser tensor...")
    tensor = create_IMUPoser_tensor(synced_dfs)

    return synced_dfs, tensor


def full_sensor_pipeline(data_dir='rawdata'):
    # Step 1: Align sensor data
    print("Aligning sensor data...")
    aligned_dfs = align_all_sensor_data(data_dir)

    # Step 2: Process aligned data
    df_names = ['Phone', 'Earbud', 'Left Watch', 'Right Watch']
    synced_dfs, tensor = process_aligned_sensor_data(aligned_dfs, df_names)

    return synced_dfs, tensor


