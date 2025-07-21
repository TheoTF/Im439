import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, dataset_config):

        self.input_df = dataset_config.input_df
        self.window_size = dataset_config.window_size
        self.step_size = dataset_config.step_size
        self.max_time_gap = dataset_config.max_time_gap
        self.sensor_column = dataset_config.sensor_column
        
        self.windows = self.get_windows()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]

    def get_windows(self):
        windows = []

        # Sort the DataFrame by Line, Trip, and Distance to ensure correct windowing
        df = self.input_df.sort_values(['Line', 'Trip']).reset_index(drop=True)
        
        for line in df['Line'].unique():
            df_line = df[df['Line'] == line]
            for trip in df_line['Trip'].unique():
                df_trip = df_line[df_line['Trip'] == trip]
                n = len(df_trip)

                for i in range(0, n - self.window_size + 1, self.step_size):
                    window = df_trip.iloc[i:i + self.window_size]
                    time_diff = window['Timestamp'].diff()#.dt.total_seconds()
                    if any(time_diff > self.max_time_gap):
                        continue
                    else:
                        windows.append(window[self.sensor_column].to_numpy())

        return windows 

if __name__ == "__main__":
    df = pd.read_parquet("df_ETL_VI_5_Maint_2143372.parquet")
    df['Trip'] = df['TripNumber'].astype(str)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    dataset = TimeSeriesDataset(df, window_size=100, step_size=50, max_time_gap=10, sensor_column='UA_Z_AL')
    print("Number of windows extracted:", len(dataset))
    print("First window data:", dataset[0])