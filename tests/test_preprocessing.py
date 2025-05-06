import unittest
import pandas as pd
import numpy as np

# Import the sensor processor classes
from preprocessing.phone_processor import PhoneSensorAligner
from preprocessing.watch_processor import WatchSensorAligner
from preprocessing.earbuds_processor import EarbudSensorAligner
from preprocessing.trim_timestamps import time_to_ms, trim_dataframes
from preprocessing.imuDataPipeline import is_accelerometer_file, is_gyroscope_file, find_sensor_files, fix_gyroscope_columns, correct_gyroscope_bias


class TestPhoneSensorAligner(unittest.TestCase):
    """Basic unit tests for PhoneSensorAligner class"""
    def setUp(self):
        self.aligner = PhoneSensorAligner()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'time': ['2023-01-01 12:00:00', '2023-01-01 12:00:01'],
            'ax': [9.81, 19.62],  # in m/s²
            'ay': [0, 4.905],
            'az': [-9.81, -14.715],
            'wx': [0.5, 1.0],  # in rad/s
            'wy': [1.5, 2.0],
            'wz': [0.1, 0.2]
        })
    
    def test_convert_ms2_to_g(self):
        """Test conversion from m/s² to g"""
        g_value = self.aligner.convert_ms2_to_g(9.81)
        self.assertEqual(g_value, 1.0)
    
    def test_convert_rads_to_degs(self):
        """Test conversion from rad/s to deg/s"""
        deg_value = self.aligner.convert_radians_to_degs(np.pi)
        self.assertEqual(deg_value, 180.0)
    
    def test_align_sensor_data(self):
        """Test the main data alignment process"""
        result = self.aligner.align_sensor_data(self.sample_data)
        self.assertIsNotNone(result)
        expected_columns = ['timestamp', 'x-axis (g)', 'y-axis (g)', 'z-axis (g)', 
                           'x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
        for col in expected_columns:
            self.assertIn(col, result.columns) # type: ignore


class TestWatchSensorAligner(unittest.TestCase):
    """Basic unit tests for WatchSensorAligner class"""
    
    def setUp(self):
        self.aligner = WatchSensorAligner()
        
        # Create sample accelerometer data
        self.accel_data = pd.DataFrame({
            'timestamp (+0000)': ['2023-01-01T12:00:00.000+0000', '2023-01-01T12:00:01.000+0000'],
            'x-axis (g)': [0.1, 0.2],
            'y-axis (g)': [0.3, 0.4],
            'z-axis (g)': [0.5, 0.6],
            'epoc (ms)': [1672574400000, 1672574401000]
        })
        
        # Create sample gyroscope data
        self.gyro_data = pd.DataFrame({
            'timestamp (+0000)': ['2023-01-01T12:00:00.000+0000', '2023-01-01T12:00:01.000+0000'],
            'x-axis (deg/s)': [10, 20],
            'y-axis (deg/s)': [30, 40],
            'z-axis (deg/s)': [50, 60],
            'epoc (ms)': [1672574400000, 1672574401000]
        })
    
    def test_process_timestamp(self):
        """Test timestamp processing"""
        result = self.aligner.process_timestamp(self.accel_data)
        self.assertIsNotNone(result)
        self.assertIn('timestamp', result.columns) # type: ignore
    
    def test_align_sensor_data(self):
        """Test sensor data alignment and merging"""
        result = self.aligner.align_sensor_data(self.accel_data, self.gyro_data)
        self.assertIsNotNone(result)
        
        # Check if the result has expected columns
        expected_cols = ['timestamp', 'x-axis (g)', 'y-axis (g)', 'z-axis (g)',
                        'x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
        for col in expected_cols:
            self.assertIn(col, result.columns)


class TestEarbudSensorAligner(unittest.TestCase):
    """Basic unit tests for EarbudSensorAligner class"""
    
    def setUp(self):
        self.aligner = EarbudSensorAligner()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': ['2023-01-01T12:00:00.000+0000', '2023-01-01T12:00:01.000+0000'],
            'x': [0.1, 0.2],
            'y': [0.3, 0.4],
            'z': [0.5, 0.6],
            'packetIndex': [1000, 1001]
        })
    
    def test_align_sensor_data(self):
        """Test sensor data alignment"""
        result = self.aligner.align_sensor_data(self.sample_data)
        self.assertIsNotNone(result)
        self.assertIn('time', result.columns) # type: ignore
        self.assertNotIn('packetIndex', result.columns) # type: ignore


class TestIntegration(unittest.TestCase):
    """Simple integration test for all sensor processors"""
    
    def setUp(self):
        # Initialize sensor processors
        self.phone_aligner = PhoneSensorAligner()
        self.watch_aligner = WatchSensorAligner()
        self.earbud_aligner = EarbudSensorAligner()
        
        # Sample data for phone
        self.phone_data = pd.DataFrame({
            'time': ['2023-01-01 12:00:00'],
            'ax': [9.81], 'ay': [0], 'az': [-9.81],
            'wx': [0.5], 'wy': [1.5], 'wz': [0.1]
        })
        
        # Sample data for watch
        self.watch_accel = pd.DataFrame({
            'timestamp (+0000)': ['2023-01-01T12:00:00.000+0000'],
            'x-axis (g)': [0.1], 'y-axis (g)': [0.3], 'z-axis (g)': [0.5],
            'epoc (ms)': [1672574400000]
        })
        
        self.watch_gyro = pd.DataFrame({
            'timestamp (+0000)': ['2023-01-01T12:00:00.000+0000'],
            'x-axis (deg/s)': [10], 'y-axis (deg/s)': [30], 'z-axis (deg/s)': [50],
            'epoc (ms)': [1672574400000]
        })
        
        # Sample data for earbuds
        self.earbud_data = pd.DataFrame({
            'timestamp': ['2023-01-01T12:00:00.000+0000'],
            'x': [0.1], 'y': [0.3], 'z': [0.5],
            'packetIndex': [1000]
        })
    
    def test_all_processors(self):
        """Test that all processors can handle their data without errors"""
        # Process data from each device
        phone_result = self.phone_aligner.align_sensor_data(self.phone_data)
        
        watch_accel_processed = self.watch_aligner.process_timestamp(self.watch_accel)
        watch_gyro_processed = self.watch_aligner.process_timestamp(self.watch_gyro)
        watch_result = self.watch_aligner.align_sensor_data(watch_accel_processed, watch_gyro_processed)
        
        earbud_result = self.earbud_aligner.align_sensor_data(self.earbud_data)
        
        # Simple verification
        self.assertIsNotNone(phone_result)
        self.assertIsNotNone(watch_result)
        self.assertIsNotNone(earbud_result)


class TestFileDetection(unittest.TestCase):
    """Tests for sensor file detection functions"""
    
    def setUp(self):
        import tempfile
        import os
        
        # Create temp directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock sensor files
        self.phone_path = os.path.join(self.temp_dir.name, "phone_data.csv")
        self.earbud_path = os.path.join(self.temp_dir.name, "esense_data.csv")
        self.left_accel_path = os.path.join(self.temp_dir.name, "left_accelerometer.csv")
        self.left_gyro_path = os.path.join(self.temp_dir.name, "left_gyroscope.csv")
        
        # Create empty files
        for path in [self.phone_path, self.earbud_path, self.left_accel_path, self.left_gyro_path]:
            with open(path, 'w') as f:
                f.write("timestamp,x,y,z\n")
                f.write("2023-01-01T12:00:00,0,0,0\n")
    
    def tearDown(self):
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_is_accelerometer_file(self):
        """Test accelerometer file detection"""
        # File with accelerometer in the name should be detected correctly
        self.assertTrue(is_accelerometer_file(self.left_accel_path))
        
        # Phone file without accelerometer in name should not be detected as accelerometer
        self.assertFalse(is_accelerometer_file(self.phone_path))
    
    def test_is_gyroscope_file(self):
        """Test gyroscope file detection"""
        # File with gyroscope in the name should be detected correctly
        self.assertTrue(is_gyroscope_file(self.left_gyro_path))
        
        # Phone file without gyroscope in name should not be detected as gyroscope
        self.assertFalse(is_gyroscope_file(self.phone_path))
    
    def test_find_sensor_files(self):
        """Test detecting sensor files in a directory"""
        # Test with the temp directory containing our mock files
        data_files = find_sensor_files(self.temp_dir.name)
        
        # Check that the right files were identified
        self.assertIsNotNone(data_files.get('earbud'))
        self.assertIsNotNone(data_files.get('left_accel'))
        self.assertIsNotNone(data_files.get('left_gyro'))


class TestTimestampProcessing(unittest.TestCase):
    """Tests for timestamp processing functions"""
    
    def test_time_to_ms(self):
        """Test conversion of time strings to milliseconds"""
        # Test ISO format with milliseconds
        self.assertEqual(time_to_ms("2023-01-01T12:34:56.789"), 45296789)
        
        # Test time-only format with milliseconds
        self.assertEqual(time_to_ms("12:34:56:789"), 45296789)
        
        # Test time-only format without milliseconds
        self.assertEqual(time_to_ms("12:34:56"), 45296000)
        
        # Test error handling for invalid format
        self.assertEqual(time_to_ms("invalid_time"), 0)
    
    def test_trim_dataframes(self):
        """Test trimming dataframes to common time range"""
        # Create sample dataframes with different time ranges
        df1 = pd.DataFrame({
            'timestamp': ['12:00:00', '12:00:01', '12:00:02'],
            'time_ms': [43200000, 43201000, 43202000],
            'value': [1, 2, 3]
        })
        
        df2 = pd.DataFrame({
            'timestamp': ['12:00:01', '12:00:02', '12:00:03'],
            'time_ms': [43201000, 43202000, 43203000],
            'value': [10, 20, 30]
        })
        
        # Trim dataframes
        trimmed_dfs = trim_dataframes([df1, df2], ['df1', 'df2'])
        
        # Check that dataframes were trimmed correctly
        # This is a simple test that doesn't fully validate the function's logic
        self.assertIsNotNone(trimmed_dfs[0])
        self.assertIsNotNone(trimmed_dfs[1])


class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the full pipeline"""
    
    def setUp(self):
        import tempfile
        import os
        
        # Create temp directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock phone data
        self.phone_path = os.path.join(self.temp_dir.name, "phone_data.csv")
        with open(self.phone_path, 'w') as f:
            f.write("time,ax,ay,az,wx,wy,wz\n")
            f.write("2023-01-01 12:00:00,9.81,0,-9.81,0.5,1.5,0.1\n")
            f.write("2023-01-01 12:00:01,9.81,0,-9.81,0.5,1.5,0.1\n")
    
    def tearDown(self):
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_fix_gyroscope_columns(self):
        """Test fixing gyroscope column headers"""
        # Create sample dataframe with incorrect column headers
        df = pd.DataFrame({
            'timestamp': ['2023-01-01 12:00:00'],
            'x-axis (g)': [0.1],
            'y-axis (g)': [0.2],
            'z-axis (g)': [0.3]
        })
        
        # Fix column headers
        fixed_df = fix_gyroscope_columns(df)
        
        # Check that column headers were fixed
        self.assertIn('x-axis (deg/s)', fixed_df.columns) # type: ignore
        self.assertIn('y-axis (deg/s)', fixed_df.columns) # type: ignore
        self.assertIn('z-axis (deg/s)', fixed_df.columns) # type: ignore
    
    def test_correct_gyroscope_bias(self):
        """Test gyroscope bias correction"""
        # Create a mock aligned dataframe with gyroscope data
        phone_df = pd.DataFrame({
            'timestamp': ['2023-01-01 12:00:00'],
            'x-axis (g)': [1.0],
            'y-axis (g)': [0.0],
            'z-axis (g)': [-1.0],
            'x-axis (deg/s)': [0.5],
            'y-axis (deg/s)': [1.5],
            'z-axis (deg/s)': [0.1]
        })
        
        # Apply bias correction
        corrected_dfs = correct_gyroscope_bias([phone_df, None, None, None])
        
        # Check that bias was corrected - only check that it returned something
        self.assertIsNotNone(corrected_dfs[0])
        self.assertIsNone(corrected_dfs[1])  # Should still be None


if __name__ == '__main__':
    unittest.main()