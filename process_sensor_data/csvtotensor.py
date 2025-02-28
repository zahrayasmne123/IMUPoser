import numpy as np
import torch # type: ignore

def validate_input_data(dfs, device_names):
    print("\nValidating input data:")
    print("-" * 50)
    
    # Filter out None values from dfs and device_names
    valid_dfs = []
    valid_device_names = []
    for df, name in zip(dfs, device_names):
        if df is not None:
            valid_dfs.append(df)
            valid_device_names.append(name)

    if not valid_dfs:
        raise ValueError("No valid dataframes to process")
    
    

    expected_cols = ['timestamp'] + [f'{ax}-axis (m/s^2)' for ax in ['x', 'y', 'z']] + \
                   [f'R{i}{j}' for i in range(3) for j in range(3)]
    
    for df, name in zip(dfs, device_names):
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} is missing columns: {missing_cols}")
    
    rows = [len(df) for df in dfs]
    if len(set(rows)) != 1:
        raise ValueError(f"Inconsistent number of rows: {dict(zip(device_names, rows))}")
    
    base_timestamps = dfs[0]['timestamp'].values
    for df, name in zip(dfs[1:], device_names[1:]):
        if not np.array_equal(base_timestamps, df['timestamp'].values):
            raise ValueError(f"Timestamps don't match between {device_names[0]} and {name}")
    
    print("✓ All input data validated successfully")
    print(f"✓ Number of frames: {rows[0]}")
    return rows[0]

def create_device_tensor(df):
    acc_values = df[[f'{ax}-axis (m/s^2)' for ax in ['x', 'y', 'z']]].values
    rot_values = df[[f'R{i}{j}' for i in range(3) for j in range(3)]].values
    return np.concatenate([acc_values, rot_values], axis=1)

def create_IMUPoser_tensor(dfs_list):
    device_names = ['phone', 'left_watch', 'right_watch', 'earbuds']
    n_frames = validate_input_data(dfs_list, device_names)
    
    tensor_data = np.zeros((n_frames, 60))
    
    # Fixed device positions with phone always on left
    device_positions = {
        'phone': 0,  # Left phone position
        'left_watch': 1,
        'left_headphone': 2,
        'right_watch': 4
    }
    
    # Process phone data (always in left position)
    phone_data = create_device_tensor(dfs_list[0])
    tensor_data[:, device_positions['phone']*12:(device_positions['phone']+1)*12] = phone_data
    
    # Process watch data
    left_watch_data = create_device_tensor(dfs_list[1])
    right_watch_data = create_device_tensor(dfs_list[2])
    tensor_data[:, device_positions['left_watch']*12:(device_positions['left_watch']+1)*12] = left_watch_data
    tensor_data[:, device_positions['right_watch']*12:(device_positions['right_watch']+1)*12] = right_watch_data
    
    # Process earbuds data
    earbuds_data = create_device_tensor(dfs_list[3])
    tensor_data[:, device_positions['left_headphone']*12:(device_positions['left_headphone']+1)*12] = earbuds_data
    
    tensor = torch.from_numpy(tensor_data).float()
    
    print("\nValidating output tensor:")
    print("-" * 50)
    print(f"✓ Tensor shape: {tensor.shape}")
    print(f"✓ Expected shape: ({n_frames}, 60)")
    
    # Save tensor
    torch.save({'imu_data': tensor}, 'rawdata/processed_dataset/IMUPoser_data.pt')
    print("\nSaved tensor to rawdata/processed_dataset/IMUPoser_data.pt")
    return tensor