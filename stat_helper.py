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


'''
Use 1.5 x IQR rule
'''
def get_outliers(df):
    first_quartiles = df.quantiles(q = 0.25)
    third_quartiles = df.quantiles(q = 0.75)
    iqr = third_quartiles - first_quartiles
    outliers = df.loc[(df < first_quartiles - 1.5 * iqr) & (df > third_quartiles + 1.5 * iqr)]
    return outliers


def get_stats_list(df):
    return [calc_mean(df),calc_median(df),calc_std(df),calc_max(df),calc_min(df)]
