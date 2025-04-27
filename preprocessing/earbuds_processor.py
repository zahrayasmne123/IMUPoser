class EarbudSensorAligner:
   def __init__(self):
       self.timestamp_columns = ['timestamp (+0000)', 'timestamp']
       self.epoch_columns = ['packetIndex']

   def align_sensor_data(self, df):
       if df is None:
        return None

       if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})

       # Find and process timestamp column if it exists
       timestamp_col = next((col for col in self.timestamp_columns if col in df.columns), None)

       if timestamp_col:
           try:
               df['time'] = df[timestamp_col].apply(
                   lambda ts: ts.split('T')[1].split('.')[0] if 'T' in str(ts) else str(ts)
               )
               df = df.drop(columns=[timestamp_col])
               cols = ['time'] + [col for col in df.columns if col != 'time']
               df = df[cols]
           except Exception as e:
               print(f"Error processing timestamps: {e}")
               return None

       # Remove epoch columns
       epoch_cols = [col for col in self.epoch_columns if col in df.columns]
       if epoch_cols:
           df = df.drop(columns=epoch_cols)

       return df
   
   def validate_data(self, df):
        timestamp_col = next((col for col in self.timestamp_columns if col in df.columns), None)
        if not timestamp_col:
            print(f"Earbuds Missing required timestamp column. Expected one of: {self.timestamp_columns}")
            return False
        return True