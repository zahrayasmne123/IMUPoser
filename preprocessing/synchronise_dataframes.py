import pandas as pd
import numpy as np
from scipy import interpolate


"""SYNCHRONISE DATAFRAMES: Data synchronisation for data collected from across multiple devices. It 
aligns timestamped data, handles different timestamp formats and interpolating sensor values to a
consistent sampling rate. This means data from the different sources can be compared. 

1. Extract time: Checks if timestamp has a 'T' string and if so splits here to
 takes the second part (the time)
2. Prepare Timestamp: Use extract time function to find time and convert timestamps to datetime
3. Find Common Time Frame: Find common time range and generate target timestamps
4. Resample Dataframes:Transform  original data with different sampling rates into consistent data
aligned to timestamps. For each numeric column: a linear interpolation function is made and values at each target 
timestamp are calculated. Missing values are replced them with mean values
5. Sync Dataframes: Puts previous functions together in a pipeline to synchronise dataframes
"""

#Split time on char "T"
def extract_time(timestamp):
    if 'T' in timestamp:
        return timestamp.split('T')[1].replace('.', ':').split('+')[0]
    return timestamp


def standardise_timestamps(dfs):
    for individual_df in dfs:
        if 'timestamp' in individual_df.columns:
            individual_df['timestamp'] = individual_df['timestamp'].apply(extract_time)
    
    # Convert timestamps to datetime
    for individual_df in dfs:
        try:
            individual_df['timestamp'] = pd.to_datetime( 
                '2025-02-25 ' + individual_df['timestamp'].str.replace(':', '.', n=3), 
                format='%Y-%m-%d %H.%M.%S.%f',
                errors='coerce'
            )
        except Exception:            
            print("Error converting timestamps")
            print(f"Sample timestamps: {individual_df['timestamp'].head().tolist()}")
    
    return dfs

def find_common_time_frame(dfs):
    # Time duration is the latest start time and earliest end time 
    start_time = max(df['timestamp'].min() for df in dfs).round('ms')
    end_time = min(df['timestamp'].max() for df in dfs)
    duration_seconds = (end_time - start_time).total_seconds()
    
    print("\nCommon time range:") #For debugging 
    print(f"Start: {start_time}")
    print(f"End: {end_time}")
    
    # Generate target timestamps at 30
    num_frames = int(duration_seconds * 30)
    target_timestamps = [
        start_time + pd.Timedelta(seconds=i/30) 
        for i in range(num_frames)
    ]
    
    return start_time, target_timestamps, num_frames


def resample_dataframes(dfs, device_names, start_time, target_timestamps, num_frames):
    processed_dfs = []
    
    # Process each dataframe
    for df, name in zip(dfs, device_names):
        print(f"\nProcessing {name}...")
        
        # Remove any rows outside the common time range
        valid_timeframe_df = df[(df['timestamp'] >= start_time) & 
                         (df['timestamp'] <= target_timestamps[-1])].copy()
        
        resampled_data = {'timestamp': target_timestamps}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        #time relative to start for interpolation
        x = (valid_timeframe_df['timestamp'] - start_time).dt.total_seconds() #converts the time difference to seconds 
        x_new = (pd.Series(target_timestamps) - start_time).dt.total_seconds()
        
        for col in numeric_columns:
            y = valid_timeframe_df[col].values
            
            try:
                # interpolation function
                f = interpolate.interp1d(
                    x, y, kind='linear',
                    bounds_error=False, fill_value='extrapolate' # type: ignore
                )
                
                # Interpolate values and handle NaNs
                interpolated_values = f(x_new)

        
                mean_value = np.nanmean(interpolated_values)
                resampled_data[col] = np.nan_to_num(interpolated_values, nan=mean_value) #missing values = mean
            
            except Exception as e:
                print(f"Error interpolating {col} for {name}: {e}")
        
        # New resampled dataframe
        fps30_resampled = pd.DataFrame(resampled_data)
        fps30_resampled['timestamp'] = fps30_resampled['timestamp'].dt.strftime('%H:%M:%S:%f').str[:-3]
        processed_dfs.append(fps30_resampled)
    
    return processed_dfs



def sync_dataframes(dfs, device_names):
    dfs = standardise_timestamps(dfs)
    
    start_time, target_timestamps, num_frames = find_common_time_frame(dfs)
    
    processed_dfs = resample_dataframes(dfs, device_names, start_time, 
                                       target_timestamps, num_frames)

    return processed_dfs


