def time_to_ms(time_str):
    """Convert any time string format to milliseconds since midnight."""
    try:
        # Extract just the time part (hours:minutes:seconds)
        if 'T' in time_str:
            # ISO format like "2025-02-25T11:25:23.415"
            # Extract the hours, minutes, seconds
            time_part = time_str.split('T')[1]
            if '.' in time_part:
                hours, minutes, seconds = map(int, time_part.split('.')[0].split(':'))
                ms = int(time_part.split('.')[1][:3])
            else:
                hours, minutes, seconds = map(int, time_part.split(':'))
                ms = 0
        elif ':' in time_str:
            # Format like "11:25:29:315" (phone/earbud format)
            parts = time_str.split(':')
            if len(parts) == 4:
                hours, minutes, seconds, ms = map(int, parts)
            else:
                hours, minutes, seconds = map(int, parts)
                ms = 0
        else:
            # Unknown format
            return 0
            
        # Calculate milliseconds since midnight (ignoring the date part)
        return ((hours * 60 + minutes) * 60 + seconds) * 1000 + ms
        
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}")
        return 0
    
def trim_dataframes(dataframes, df_names):
    for i, df in enumerate(dataframes):
        print(f"DataFrame {df_names[i]} columns: {df.columns.tolist()}")
        print(f"DataFrame {df_names[i]} has {len(df)} rows")
    
    # Standardize timestamp column names
    for i, df in enumerate(dataframes):
        if 'timestamp' not in df.columns and 'time' in df.columns:
            dataframes[i] = df.rename(columns={'time': 'timestamp'})
            print(f"Renamed 'time' to 'timestamp' in {df_names[i]}")
    
    # Print first few timestamps to debug
    for i, df in enumerate(dataframes):
        if 'timestamp' in df.columns and len(df) > 0:
            print(f"{df_names[i]} first timestamps: {df['timestamp'].head(3).tolist()}")
    
    # Add milliseconds for comparison - with better error handling
    for i, df in enumerate(dataframes):
        try:
            if 'timestamp' in df.columns:
                # Print timestamp format for debugging
                first_ts = df['timestamp'].iloc[0] if len(df) > 0 else "No data"
                print(f"{df_names[i]} first timestamp format: {first_ts}")
                
                # Add a safer ms conversion
                dataframes[i]['time_ms'] = df['timestamp'].apply(time_to_ms)
        except Exception as e:
            print(f"Error processing timestamps for {df_names[i]}: {e}")
            # Use a simple index-based timestamp as fallback
            dataframes[i]['time_ms'] = range(len(df))
    
    # Find common time range
    min_times = []
    max_times = []
    
    for i, df in enumerate(dataframes):
        try:
            if 'timestamp' in df.columns:
                # Print timestamp format for debugging
                first_ts = df['timestamp'].iloc[0] if len(df) > 0 else "No data"
                print(f"{df_names[i]} first timestamp format: {first_ts}")
                
                # Use the improved time_to_ms function for all devices
                dataframes[i]['time_ms'] = df['timestamp'].apply(time_to_ms)
        except Exception as e:
            print(f"Error processing timestamps for {df_names[i]}: {e}")
            # Use a simple index-based timestamp as fallback
            dataframes[i]['time_ms'] = range(len(df))
    
    if not min_times or not max_times:
        print("No valid time ranges found")
        return dataframes  # Return original dataframes instead of empty ones
    
    latest_start_ms = max(min_times)
    earliest_end_ms = min(max_times)
    
    print(f"\nCommon time range: {latest_start_ms} - {earliest_end_ms}")
    
    if latest_start_ms > earliest_end_ms:
        print("WARNING: No common time range found - returning original dataframes")
        return dataframes
    
    # Trim dataframes
    trimmed_dfs = []
    for df, name in zip(dataframes, df_names):
        if len(df) == 0 or 'time_ms' not in df.columns:
            print(f"Skipping empty or invalid dataframe: {name}")
            trimmed_dfs.append(df)
            continue
            
        trimmed_df = df[
            (df['time_ms'] >= latest_start_ms) & 
            (df['time_ms'] <= earliest_end_ms)
        ].copy()
        
        print(f"\nProcessed {name}:")
        print(f"Original rows: {len(df)}")
        print(f"Trimmed rows: {len(trimmed_df)}")
        
        if not trimmed_df.empty:
            print(f"Start time: {trimmed_df['timestamp'].iloc[0]}")
            print(f"End time: {trimmed_df['timestamp'].iloc[-1]}")
        else:
            print(f"Warning: Trimmed dataframe for {name} is empty")
        
        trimmed_dfs.append(trimmed_df)
    
    return trimmed_dfs