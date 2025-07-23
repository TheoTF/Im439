import pandas as pd
from torch.utils.data import Dataset
from config import GlobalConfig
class TimeSeriesDataset(Dataset):
    def __init__(self, global_config: GlobalConfig, input_df):

        self.input_df = input_df
        self.window_size = global_config.dataset_config.window_size
        self.step_size = global_config.dataset_config.step_size
        self.max_time_gap = global_config.dataset_config.max_time_gap
        self.sensor_column = global_config.dataset_config.sensor_column

        self.windows_time = self.get_windows()

    def __len__(self):
        return len(self.windows_time)

    def __getitem__(self, index):
        return self.windows_time[index]

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
                    time_diff = window['Timestamp'].diff().dt.total_seconds()
                    if any(time_diff > self.max_time_gap):
                        continue
                    else:
                        windows.append(window[self.sensor_column].to_numpy())

        return windows 

if __name__ == "__main__":
    #TODO: Adjust this example
    df = pd.read_parquet("df_ETL_VI_5_Maint_2143372.parquet")
    df['Trip'] = df['TripNumber'].astype(str)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    dataset = TimeSeriesDataset(df, window_size=100, step_size=50, max_time_gap=10, sensor_column='UA_Z_AL')
    print("Number of windows extracted:", len(dataset))
    print("First window data:", dataset[0])