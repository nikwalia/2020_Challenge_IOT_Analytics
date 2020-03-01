import numpy as np
import pandas as pd

import h5py
import os
import re
from dateutil.parser import parse
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from datetime import datetime, date, time, timedelta
import data_cleaning

GLOBAL_SAMPLING_FREQ = 10
GLOBAL_DAILY_SAMPLES = 24 * 60 * 6 * 2
GLOBAL_TIME_RANGE = []
for i in range(GLOBAL_DAILY_SAMPLES):

    secs = i * 5
    hours = int(secs / (3600))
    remainder = int(secs - hours * 3600)
    mins = int(remainder / (60))
    remainder = int(remainder - mins * 60)
    secs = int(remainder)
    GLOBAL_TIME_RANGE.append(time(hours, mins, secs))

GLOBAL_ZERO_DATE = datetime(1, 1, 1, 0, 0, 0, 0)


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
        time_values.append(date + timedelta(seconds = i * time_interval))

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
    if '/' in split[0]:
        date = split[0].split('/')
    else:
        date = split[0].split('-')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    time = split[1].split(':')
    hour = int(time[0])
    minute = int(time[1])
    second = int(time[2].split('.')[0])
    millisecond = int(time[2].split('.')[1]) * 1000
    return datetime(year, month, day, hour, minute, second, millisecond)


'''
Resamples data points data.

return resampled data. May be upsampled or downsampled
'''
def resample(data, original_freq, new_freq):
    if original_freq == new_freq:
        return data
    
    num_new_points = int(round(len(data) * new_freq / original_freq))
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
def find_closest_timestamp(timestamp):
    min_err = abs(datetime.combine(GLOBAL_ZERO_DATE, timestamp.time()) - datetime.combine(GLOBAL_ZERO_DATE, GLOBAL_TIME_RANGE[0]))
    best_timestamp = GLOBAL_TIME_RANGE[0]
    
    for possible_timestamp in GLOBAL_TIME_RANGE:
        potential_err = abs(datetime.combine(GLOBAL_ZERO_DATE, timestamp.time()) - datetime.combine(GLOBAL_ZERO_DATE, possible_timestamp))
        
        if potential_err < min_err:
            min_err = potential_err
            best_timestamp = possible_timestamp
#             print(min_err, best_timestamp)
    
#     print(min_err, best_timestamp)
    return datetime.combine(timestamp.date(), best_timestamp)


def generate_time_range(begin, end):
    time_vals = []
    
#     print(end_time)
    
    while begin <= end:
        time_vals.append(begin.time())
        begin = begin + timedelta(seconds = 5)
        
    return np.array(time_vals)


def get_channel_data(channel_id):
    channel_df = pd.DataFrame(index = GLOBAL_TIME_RANGE)
    # channel_df['time'] = GLOBAL_TIME_RANGE
    # channel_df.set_index('time')

    # print(channel_df)

    files = os.listdir('./competitionfiles')

    unique_dates = []
    for file in files:
        date_str = re.findall('(\d+)', file)[0]
        if date_str not in unique_dates:
            unique_dates.append(date_str)

    for date_str in unique_dates:
        files_with_date = []
        
        formatted_date = date_str[0:4] + '-' + date_str[4:6] + '-' + date_str[6:]

        date_series = pd.Series(data = np.NAN, index = GLOBAL_TIME_RANGE)
        for file in files:
            if date_str in file:
                files_with_date.append(file)

        for file in files_with_date:
            print(file)
            f = h5py.File('competitionfiles/' + file)
            if channel_id not in list(f['DYNAMIC DATA'].keys()):
                continue
            readings = np.array(f['DYNAMIC DATA'][channel_id]['MEASURED'])
            if len(readings) == 0:
                continue
                
            data_for_date = {}

            data_for_date['start_time'] = convert_timestamp(str(f['DYNAMIC DATA'].attrs['FIRST ACQ TIMESTAMP']))
            
            data_for_date['end_time'] = convert_timestamp(str(f['DYNAMIC DATA'].attrs['LAST ACQ TIMESTAMP']))
            

            data_for_date['freq'] = float(f['DYNAMIC DATA'][channel_id].attrs['SAMPLE RATE'])
            data_for_date['sensor_readings'] = np.array(f['DYNAMIC DATA'][channel_id]['MEASURED']).astype(np.float64)

            # re-sample data
            data_for_date['sensor_readings'] = resample(
                                                        data_for_date['sensor_readings'],
                                                        data_for_date['freq'],
                                                        GLOBAL_SAMPLING_FREQ
                                                        )
            
            data_for_date['sensor_readings'] = data_cleaning.down_sample(data_for_date['sensor_readings'])
            data_for_date['freq'] = 0.2
            
            data_for_date['start_time'] = find_closest_timestamp(data_for_date['start_time'])
            data_for_date['end_time'] = find_closest_timestamp(data_for_date['end_time'])

            insert_timestamps = generate_time_range(data_for_date['start_time'], data_for_date['end_time'])

            if data_for_date['start_time'].date() < data_for_date['end_time'].date():
                same_date_loc = np.where(insert_timestamps > data_for_date['end_time'].time())
                insert_timestamps = insert_timestamps[same_date_loc]
                data_for_date['sensor_readings'] = data_for_date['sensor_readings'][same_date_loc]

            if len(insert_timestamps) < len(data_for_date['sensor_readings']):
                data_for_date['sensor_readings'] = data_for_date['sensor_readings'][:-1]
            elif len(insert_timestamps) > len(data_for_date['sensor_readings']):
                insert_timestamps = insert_timestamps[:-1]

            date_series.loc[insert_timestamps] = data_for_date['sensor_readings']
            f.close()
            
        # print(date_series.mean())
        date_series.index = channel_df.index
        channel_df[parse(formatted_date).date()] = date_series
        print(channel_df.mean()[-1])
        print(len(channel_df.columns))
    return channel_df