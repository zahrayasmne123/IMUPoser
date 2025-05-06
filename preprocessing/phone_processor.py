import pandas as pd
import numpy as np

"""PHONE SENSOR ALIGNER: First step in pipeline, taking raw phone data and converting acceleration measurements 
from m/s^2 to g units, and angular velocity from radians per second to degrees. The class also cleans and renames columns
to more descriptive format

1: Initialise variables mappings for column names
2. Load data:  Reads CSV file loading only the expected columns
3. Convert ms2 to g: Defines gravity as 9.81 and calculates valeus in g units to 3dp 
4. Convert Radians to Degrees: Convert gyroscope valued from rad/s to deg/s
5. Align Sensor Data: Builds on previous functions to make a mini pipeline soley for processinh phone data"""

class PhoneSensorAligner:
    def __init__(self):
        self.IMU_COLUMNS = {  # Dictionary mapping original column names to more descriptive names
            'ax': 'x-axis (g)',
            'ay': 'y-axis (g)',
            'az': 'z-axis (g)',
            'wx': 'x-axis (deg/s)',
            'wy': 'y-axis (deg/s)',
            'wz': 'z-axis (deg/s)'
        }

        # Column indices for acceleration and gyroscope data
        self.ACCELEROMETER_COLUMNS = ['ax', 'ay', 'az']
        self.GYROSCOPE_COLUMNS = ['wx', 'wy', 'wz']
        
    def load_data(self, file_path):
        try:
            # Read expected columns 
            expected_columns = ['time'] + list(self.IMU_COLUMNS.keys())
            df = pd.read_csv(file_path, usecols=expected_columns)
            return df #return dataframe is sucessful 
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def convert_ms2_to_g(self, acceleration):
        return round(acceleration / 9.81, 3)

    def convert_radians_to_degs(self, angular_velocity):
        return round(angular_velocity * (180 / np.pi), 3)

    def align_sensor_data(self, df):
        """Main method to process and align sensor data"""
        if df is None:
            return None
        try:
            df = df.copy()
            
            # Remove any unnamed columns
            df = df.drop(columns=[col for col in df.columns if 'Unnamed:' in col], errors='ignore')
            
            # Rename time column
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            
            # Convert units
            for col in self.ACCELEROMETER_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].apply(self.convert_ms2_to_g)
            
            for col in self.GYROSCOPE_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].apply(self.convert_radians_to_degs)
            
            # Rename to final column names
            df = df.rename(columns=self.IMU_COLUMNS)
            return df
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return None