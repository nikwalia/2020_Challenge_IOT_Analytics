from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np

def calc_mean(df):
    return df.mean()

def calc_median(df):
    return df.median()

def calc_std(df):
    return df.std()

def calc_max(df):
    return df.max()

def calc_min(df):
    return df.min()

def smooth_values(df):
    filtered = pd.DataFrame()
    for channel in df.columns:
        filtered[channel] = gaussian_filter1d(df[channel].to_numpy(), 1)
    
    return filtered


def get_stats_list(df):
    return [calc_mean(df),calc_median(df),calc_std(df),calc_max(df),calc_min(df)]
