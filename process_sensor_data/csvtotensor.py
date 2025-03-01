import numpy as np
import torch
import os
from typing import List, Optional, Tuple, Sequence
import pandas as pd

def validate_input_data(dfs: Sequence[Optional[pd.DataFrame]], device_names: List[str]) -> Tuple[int, List[pd.DataFrame], List[str]]:
    print("\nValidating input data:")
    print("-" * 50)
    
    # Filter out None values from dfs and device_names
    valid_pairs = [(df, name) for df, name in zip(dfs, device_names) if df is not None]
    
    if not valid_pairs:
        raise ValueError("No valid dataframes to process")
    
    valid_dfs = [pair[0] for pair in valid_pairs]
    valid_device_names = [pair[1] for pair in valid_pairs]
    
    # Check for expected columns in each dataframe
    expected_cols = ['timestamp'] + [f'{ax}-axis (m/s^2)' for ax in ['x', 'y', 'z']] + \
                   [f'R{i}{j}' for i in range(3) for j in range(3)]
    
    for df, name in zip(valid_dfs, valid_device_names):
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} is missing columns: {missing_cols}")
    
    # Check row counts
    rows = [len(df) for df in valid_dfs]
    if len(set(rows)) != 1:
        raise ValueError(f"Inconsistent number of rows: {dict(zip(valid_device_names, rows))}")
    
    # Check timestamp consistency
    base_timestamps = valid_dfs[0]['timestamp'].values
    for df, name in zip(valid_dfs[1:], valid_device_names[1:]):
        if not np.array_equal(base_timestamps, df['timestamp'].to_numpy()): # type: ignore
            raise ValueError(f"Timestamps don't match between {valid_device_names[0]} and {name}")
    
    print("✓ All input data validated successfully")
    print(f"✓ Number of frames: {rows[0]}")
    print(f"✓ Valid devices: {valid_device_names}")
    
    return rows[0], valid_dfs, valid_device_names

def create_device_tensor(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts acceleration and rotation data from a dataframe into a numpy array.
    """
    acc_values = df[[f'{ax}-axis (m/s^2)' for ax in ['x', 'y', 'z']]].values
    rot_values = df[[f'R{i}{j}' for i in range(3) for j in range(3)]].values
    return np.concatenate([acc_values, rot_values], axis=1)

def create_IMUPoser_tensor(
    dfs_list: List[Optional[pd.DataFrame]], 
    device_names: List[str] = [],
    output_path: Optional[str] = None
) -> torch.Tensor:
    """
    Creates a tensor from IMU data for the IMUPoser model with fixed mapping.
    Based on the IMUPoser paper, the tensor has 5 positions with 12 values each.
    
    Args:
        dfs_list: List of dataframes containing IMU data (can contain None for missing devices)
        device_names: Names of devices corresponding to each dataframe
        output_path: Path to save the tensor (optional)
        
    Returns:
        PyTorch tensor containing the processed IMU data
    """
    # Set default device names if not provided
    if device_names is None:
        device_names = ["phone", "left_watch", "right_watch", "earbuds"]
    
    # Validate input data - filter out None values
    valid_dfs_list = [df for df in dfs_list if df is not None]
    valid_names = [name for df, name in zip(dfs_list, device_names) if df is not None]
    
    if not valid_dfs_list:
        raise ValueError("No valid dataframes to process")
    
    n_frames, valid_dfs, valid_device_names = validate_input_data(valid_dfs_list, valid_names)
    
    
    # Define standard device mapping according to paper
    device_positions = {
        'phone': 0,        # Phone (left position)
        'left_watch': 1,   # Left watch
        'earbuds': 2,      # Earbuds/headphones (left position)
        'right_watch': 4   # Right watch
    }
    
    # Initialize tensor with zeros (n_frames x 60)
    # IMUPoser uses 5 positions with 12 values each = 60 total
    tensor_data = np.zeros((n_frames, 60))
    
    # Process each device and place in the correct position
    for df, name in zip(valid_dfs, valid_device_names):
        position = None
        
        # Try to match the device name directly
        if name.lower() in device_positions:
            position = device_positions[name.lower()]
        # Try to match by using partial matches
        else:
            for device_key in device_positions:
                if device_key in name.lower():
                    position = device_positions[device_key]
                    break
        
        if position is not None:
            device_data = create_device_tensor(df)
            tensor_data[:, position*12:(position+1)*12] = device_data
            print(f"✓ Placed device '{name}' at position {position}")
        else:
            print(f"⚠ Warning: No position mapping found for device '{name}', skipping")
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(tensor_data).float()
    
    print("\nValidating output tensor:")
    print("-" * 50)
    print(f"✓ Tensor shape: {tensor.shape}")
    print(f"✓ Expected shape: ({n_frames}, 60)")
    
    # Save tensor if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({'imu_data': tensor}, output_path)
        print(f"\nSaved tensor to {output_path}")
    
    return tensor