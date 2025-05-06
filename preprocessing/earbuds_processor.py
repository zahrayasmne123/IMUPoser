"""EARBUDS SENSOR ALIGNER: First step in pipeline, taking raw earbud data and gathering the correct 
headers and timestamps.

1. align_sensor_data: timestamp columns, converting them into a standard format and removes epoch or packet index columns that aren't needed
This ensures that the earbud sensor data has consistent formatting before it's passed down the pipeline

2. extract_time: helper function that extracting the time portion from a timestamp by splitting on string 'T' """

class EarbudSensorAligner:
   def __init__(self):
       self.timestamp_columns = ['timestamp (+0000)', 'timestamp']
       self.epoch_columns = ['packetIndex']

   def align_sensor_data(self, df):
       if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})

       # Find and process timestamp column if it exists
       timestamp_col = next((col for col in self.timestamp_columns if col in df.columns), None)

       if timestamp_col:
            try: #Creates a new 'time' column by extracting just the time portion from the timestamp
                df['time'] = df[timestamp_col].apply(extract_time)
                df = df.drop(columns=[timestamp_col]) #Drops the initial timestamp column in df
                colum = ['time'] + [col for col in df.columns if col != 'time']
                df = df[colum]
            except Exception as e:
                print(f"Error processing timestamps: {e}")
                return None

       # Remove unecessary epoch columns
       epoch_cols = [col for col in self.epoch_columns if col in df.columns]
       if epoch_cols:
           df = df.drop(columns=epoch_cols)

       return df

def extract_time(ts):
    ts_str = str(ts)
    if 'T' in ts_str:
        return ts_str.split('T')[1].split('.')[0]
    return ts_str