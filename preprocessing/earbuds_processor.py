"""EARBUDS SENSOR ALIGNER: First step in pipeline, taking raw earbud data and gathering the correct 
headers and timestamps.

1. align_sensor_data: timestamp columns, converting them into a standard format and removes epoch or packet index columns that aren't needed
This ensures that the earbud sensor data has consistent formatting before it's passed down the pipeline

2. extract_time: helper function that extracting the time portion from a timestamp by splitting on string 'T' """

class EarbudSensorAligner:
   def __init__(self):
       self.timestamp_columns = ['timestamp (+0000)', 'timestamp']
       self.epoch_columns = ['packetIndex']

   def align_sensor_data(self, dataframe):
       if 'time' in dataframe.columns:
            dataframe = dataframe.rename(columns={'time': 'timestamp'})

       # Find and process timestamp column if it exists
       timestamp_col = next((column for column in self.timestamp_columns if column in dataframe.columns), None)

       if timestamp_col:
            try: #Creates a new 'time' column by extracting just the time portion from the timestamp
                dataframe['time'] = dataframe[timestamp_col].apply(extract_time)
                dataframe = dataframe.drop(columns=[timestamp_col]) #Drops the initial timestamp column in df
                colum = ['time'] + [col for col in dataframe.columns if col != 'time']
                dataframe = dataframe[colum]
            except Exception as e:
                print(f"Error processing timestamps: {e}")
                return None

       # Remove unecessary epoch columns
       epoch_cols = [column for column in self.epoch_columns if column in dataframe.columns]
       if epoch_cols:
           dataframe = dataframe.drop(columns=epoch_cols)

       return dataframe

def extract_time(timestamp):
    timestamp_string = str(timestamp)
    if 'T' in timestamp_string:
        return timestamp_string.split('T')[1].split('.')[0]
    return timestamp_string