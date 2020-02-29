import numpy as np
import pandas as pd

import h5py
import os

from stat_helper import *

'''
Helper to convert path of hdf file to a dataframe

path String path of location of hdf file (include ".hdf")
return Dataframe of sensor data (columns are channels, rows are data entries)
'''
def h5_to_df(path):
    h5_file = h5py.File(path,'r')
    channels = list(h5_file['DYNAMIC DATA'].keys())

    df = pd.DataFrame()

    for channel in channels:
        df[channel] = h5_file['DYNAMIC DATA'][channel]['MEASURED']

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
Helper to convert our list of statistics dataframes to a 3D np array

return a 3D numpy array
'''
def get_stats_np_arr(df_stats):
    return np.array([np.array(df) for df in df_stats])