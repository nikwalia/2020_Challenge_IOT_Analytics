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
Helper to get all data from specified folder

return list of Dataframes containing all our sensor data
'''
def get_all_data():
    df_arr = []
    files = os.listdir('./competitionfiles')

    for file in files:
        df_arr.append(h5_to_df(file))
    
    return df_arr


'''
Helper to get statistics for all dataframes

return list of Dataframes containing all the statistics
'''
def get_df_stats_list():
    df_arr = get_all_data
    stats_df_arr = []

    for df in df_arr:
        stats_df = get_df_stats(df)
        stats_df_arr.append(stats_df)

    return stats_df_arr

def get_df_stats(df):
    stat_df = pd.DataFrame()
    stat_df['Index'] = ['mean','median','std','min','max','iqr']

    