"""TRIM TIMESTAMPS: Functions to convert various time string formats to milliseconds since midnight 

1. Time to ms: Convert any time string format to milliseconds since midnight. If string has T then it is ISO format and splits on T. 
If it contains : without T then its in phone/earbud format. Otherwise = 0 (unknown). The use this to calc milliseconds since midmight

2. Trim Dataframes: Processes list of data frames and trims them to the common time range using the the latest start
time and earliest end time to find overlapping time. """

def time_to_ms(time_str):
    try:
        if 'T' in time_str:
            time_part = time_str.split('T')[1]
            if '.' in time_part:
                hours, minutes, seconds = map(int, time_part.split('.')[0].split(':'))
                ms = int(time_part.split('.')[1][:3])
        elif ':' in time_str: # (phone/earbud format)
            parts = time_str.split(':')
            if len(parts) == 4:
                hours, minutes, seconds, ms = map(int, parts)
        else:
            return 0
            
        # Calculate milliseconds since midnight (ignoring the date part)
        return ((hours * 60 + minutes) * 60 + seconds) * 1000 + ms
        
    except Exception:
        print(f"Error parsing time {time_str}")
        return 0
    


def trim_dataframes(dataframes, df_names):
    # Standardise timestamp column names by renaming 'time' columns to 'timestamp' 
    for i, df in enumerate(dataframes):
        if 'timestamp' not in df.columns and 'time' in df.columns:
            dataframes[i] = df.rename(columns={'time': 'timestamp'})
    
    # Initialise lists for common timestamps 
    min_times = []
    max_times = []
    
    for i, df in enumerate(dataframes):
        try:
            if 'timestamp' in df.columns:
                dataframes[i]['time_ms'] = df['timestamp'].apply(time_to_ms)
        except Exception as e:
            print(f"Error processing timestamps for {df_names[i]}: {e}")
    
    if not min_times or not max_times:
        print("No valid time ranges found")
        return dataframes  # Return original dataframes instead of empty ones
    
    latest_start_time = max(min_times)
    earliest_end_time = min(max_times)
    
    print(f"Common time range: {latest_start_time} - {earliest_end_time}")
    # Trim dataframes
    trimmed_dfs = []
    for df, name in zip(dataframes, df_names):
        trimmed_df = df[
            (df['time_ms'] >= latest_start_time) & 
            (df['time_ms'] <= earliest_end_time)
        ].copy()
        
        if not trimmed_df.empty:
            print(f"Start time: {trimmed_df['timestamp'].iloc[0]}")
            print(f"End time: {trimmed_df['timestamp'].iloc[-1]}")
        else:
            print("Trimmed dataframe empty")
        
        trimmed_dfs.append(trimmed_df) #add to trimmed dfs list 
    
    return trimmed_dfs