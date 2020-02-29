import numpy as np
import pandas as pd

import h5py
import os
import re
import datetime
from dateutil.parser import parse

from stat_helper import *

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