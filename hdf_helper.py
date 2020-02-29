import numpy as np
import pandas as pd

import h5py
import os
import re
import datetime
from dateutil.parser import parse
from scipy import signal

from stat_helper import *

GLOBAL_SAMPLING_FREQ = 10
GLOBAL_DAILY_SAMPLES = 24 * 60 * 60 * 10


'''
Helper to convert path of hdf file to a dataframe

path String path of location of hdf file (include ".hdf")
return Dataframe of sensor data (columns are channels, rows are data entries)
'''
def h5_to_df(path):
    h5_file = h5py.File(path,'r')
    channels = list(h5_file['DYNAMIC DATA'].keys())

    filename = r'{}'.format(path).split('\\')[-1].split('/')[-1]
    date = parse(re.findall('(\d+)', filename)[0])

    df = pd.DataFrame()

    for channel in channels:
        df[channel] = h5_file['DYNAMIC DATA'][channel]['MEASURED']

    df.datetime = date
    sample_rate = h5_file['DYNAMIC DATA'][channels[0]].attrs['SAMPLE RATE']
    time_interval = 1 / sample_rate

    df.sample_rate = sample_rate

    time_values = []
    for i in range(df.shape[0]):
        time_values.append(date + datetime.timedelta(seconds = i * time_interval))

    df['datetime'] = pd.to_datetime(time_values)
    df.index = df['datetime']
    del df['datetime']
    return df


'''
DO NOT USE UNLESS ABSOLUTELY NEEDED
Helper to get all data from specified folder

return list of Dataframes containing all our sensor data
'''
def get_all_data():
    df_arr = []
    files = os.listdir('./competitionfiles')

    for file in files:
        df_arr.append(h5_to_df('competitionfiles/' + file))
    
    return df_arr


'''
Converts a raw string from the attribute into a Python datetime object

return datetime object
'''
def convert_timestamp(str_timestamp):
    split = str_timestamp.split("'")[1].split(' ')
    date = split[0].split('/')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    time = split[1].split(':')
    hour = int(time[0])
    minute = int(time[1])
    second = int(time[2].split('.')[0])
    millisecond = int(time[2].split('.')[1]) * 1000
    return datetime.datetime(year, month, day, hour, minute, second, millisecond)


'''
Resamples data points data.

return resampled data. May be upsampled or downsampled
'''
def resample(data, original_freq, new_freq):
    if original_freq == new_freq:
        return data

    num_new_points = len(data) * new_freq / original/freq
    return signal.resample(data, num_new_points)


'''
Helper to get statistics for all dataframes

return list of Dataframes containing all the statistics
'''
def get_df_stats_list():
    stats_df_arr = []

    files = os.listdir('./competitionfiles')

    for file in files:
        stats_df_arr.append(get_df_stats(h5_to_df('competitionfiles/' + file)))

    return stats_df_arr


'''
Helper to get statistics for a single dataframe

return a df containing mean, median, std, min, and max for each channel
'''
def get_df_stats(df):
    stat_df = pd.DataFrame(columns=df.columns)

    stat_df.loc[len(stat_df)] = calc_mean(df)
    stat_df.loc[len(stat_df)] = calc_median(df)
    stat_df.loc[len(stat_df)] = calc_std(df)
    stat_df.loc[len(stat_df)] = calc_min(df)
    stat_df.loc[len(stat_df)] = calc_max(df)

    stat_df['Index'] = ['mean','median','std','min','max']

    return stat_df


'''
Helper to convert a list of dataframes to a 3D np array

return a 3D numpy array
'''
def get_stats_np_arr(list_df):
    return np.array([np.array(df) for df in list_df])


'''
Finds the closest timestamp- necessary for mapping
'''
def find_closest_timestamp(timestamp, all_timestamps):
    err = abs(timestamp - all_timestamps[0])
    best_timestamp = all_timestamps[0]
    for possible_timestamp in all_timestamps:
        if abs(timestamp - possible_timestamp) < err:
            err = abs(timetamp - possible_timestamp)
            best_timestamp = possible_timestamp
    return best_timestamp


def get_channel_data(channel_id):
    channel_df = pd.DataFrame()
    files = os.listdir('./competitionfiles')

    unique_dates = []
    for file in files:
        date = parse(re.findall('(\d+)', filename)[0])
        if date not in unique_dates:
            unique_dates.append(date)


    for date in unique_dates:
        files_with_date = []

        formatted_date = date[0:4] + '-' + date[4:6] + '-' + date[6:]

        times = pd.date_range(formatted_date, periods = GLOBAL_DAILY_SAMPLES, freq = '100L')
        date_series = pd.Series(data = np.NAN, index = times)

        for file in files:
            if date in file:
                files_with_date.append(file)
        
        data_for_date = {}
        for file in files_with_date:
            f = h5py.File('competitionfiles/' + file)
            readings = f['DYNAMIC DATA'][channel_id]['MEASURED'].to_numpy()
            if len(readings) == 0:
                continue
            
            start_time = convert_timestamp(str(f['DYNAMIC DATA'].attrs['FIRST ACQ TIMESTAMP']))
            data_for_date[file]['start_time'] = start_time

            end_time = convert_timestamp(str(f['DYNAMIC DATA'].attrs['LAST ACQ TIMESTAMP']))
            data_for_date[file]['end_time'] = end_time

            data_for_date[file]['freq'] = float(f['DYNAMIC DATA'][channel_id].attrs['SAMPLE RATE'])
            data_for_date[file]['sensor_readings'] = f['DYNAMIC DATA'][channel_id]['MEASURED'].to_numpy()

            # re-sample data
            data_for_date[file]['sensor_readings'] = downsample_data(data_for_date['sensor_readings'], data_for_date['freq'], GLOBAL_SAMPLING_FREQ)
            data_for_date[file]['freq'] = GLOBAL_SAMPLING_FREQ
        
        for key in data_for_date.keys():
            mapped_timestamp = find_closest_timestamp(data_for_date[key]['start_time'], times)
            points = data_for_date[key]['sensor_readings']
            insert_timestamps = pd.date_range(mapped_timestamp, period = len(points), freq = '100L')
            date_series.loc[insert_timestamps] = points

        channel_df[datetime.datetime(formatted_date)] = date_series

        
    return channel_df