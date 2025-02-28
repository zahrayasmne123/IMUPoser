import pandas as pd
from pathlib import Path

def load_imu_data(data_dir='.'):
    """
    Load IMU data from CSV files and prepare for model input
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        List of pandas DataFrames with IMU data
    """
    # Define file paths
    data_dir = Path(data_dir)
    phone_path = data_dir / 'data/mobileposer_phone_processed.csv'
    left_watch_path = data_dir / 'data/mobileposer_left_watch_processed.csv'
    right_watch_path = data_dir / 'data/mobileposer_right_watch_processed.csv'
    earbuds_path = data_dir / 'data/mobileposer_earbuds_processed.csv'
    
    # Load data
    print(f"Loading IMU data from {data_dir}")
    phone_df = pd.read_csv(phone_path)
    left_watch_df = pd.read_csv(left_watch_path)
    right_watch_df = pd.read_csv(right_watch_path)
    earbuds_df = pd.read_csv(earbuds_path)
    
    print(f"Phone data shape: {phone_df.shape}")
    print(f"Left watch data shape: {left_watch_df.shape}")
    print(f"Right watch data shape: {right_watch_df.shape}")
    print(f"Earbuds data shape: {earbuds_df.shape}")
    
    return [phone_df, left_watch_df, right_watch_df, earbuds_df]

def process_imu_data(dfs_list):
    """
    Process IMU data using the MobilePoser tensor creation function
    
    Args:
        dfs_list: List of pandas DataFrames with IMU data
        
    Returns:
        Tensor ready for model input
    """
    # Import the tensor creation function
    from csvtotensor import create_mobileposer_tensor
    
    # Process the data
    tensor = create_mobileposer_tensor(dfs_list)
    
    print(f"Created tensor with shape: {tensor.shape}")
    return tensor

if __name__ == "__main__":
    # Example usage
    dfs = load_imu_data()
    tensor = process_imu_data(dfs)
    print("Saved tensor to mobileposer_data.pt")