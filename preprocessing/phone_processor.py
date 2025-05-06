import pandas as pd
import numpy as np

class PhoneSensorAligner:
    """
    A class to handle phone sensor data adjustments including unit conversions
    and column renaming for IMU (accelerometer and gyroscope) data.
    """
    
    def __init__(self):
        self.STANDARD_GRAVITY = 9.81  # m/s²
        # Define the expected input column names and their corresponding output names
        self.IMU_COLUMNS = {
            'ax': 'x-axis (g)',
            'ay': 'y-axis (g)',
            'az': 'z-axis (g)',
            'wx': 'x-axis (deg/s)',
            'wy': 'y-axis (deg/s)',
            'wz': 'z-axis (deg/s)'
        }
        # Column indices for acceleration and gyroscope data (0-based indexing)
        self.ACCEL_COLS = ['ax', 'ay', 'az']
        self.GYRO_COLS = ['wx', 'wy', 'wz']
        self.DECIMAL_PLACES = 3
        
    def load_data(self, file_path):
        """Load CSV file and ensure correct column names"""
        try:
            # Print original column names before loading
            original_df = pd.read_csv(file_path)
            print("Original columns before filtering:", original_df.columns.tolist())
            
            # Read CSV with expected column names
            df = pd.read_csv(file_path, usecols=['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz'])
            
            # Print columns after loading
            print("Columns after filtering:", df.columns.tolist())
            
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def rename_time_column(self, df):
        """Rename time column to timestamp"""
        df = df.copy()
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})
        return df

    def rename_imu_columns(self, df):
        """Rename IMU columns to their standardized names"""
        return df.rename(columns=self.IMU_COLUMNS)

    def convert_ms2_to_g(self, acceleration):
        """Convert acceleration from m/s² to g"""
        return round(acceleration / self.STANDARD_GRAVITY, self.DECIMAL_PLACES)

    def convert_rads_to_degs(self, angular_velocity):
        """Convert angular velocity from rad/s to deg/s"""
        return round(angular_velocity * (180 / np.pi), self.DECIMAL_PLACES)

    def convert_imu_units(self, df):
        """Convert IMU units: acceleration to g and angular velocity to deg/s"""
        df = df.copy()
        
        # Convert accelerometer columns from m/s² to g
        for col in self.ACCEL_COLS:
            if col in df.columns:
                df[col] = df[col].apply(self.convert_ms2_to_g)
        
        # Convert gyroscope columns from rad/s to deg/s
        for col in self.GYRO_COLS:
            if col in df.columns:
                df[col] = df[col].apply(self.convert_rads_to_degs)
        
        return df

    def align_sensor_data(self, df):
        """Main method to process and align sensor data"""
        if df is None:
            return None
        
        try:
            df = self.rename_time_column(df)      # First: rename time column
            df = self.convert_imu_units(df)       # Third: convert units
            df = self.rename_imu_columns(df)      # Fourth: rename to final column names
            return df
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
