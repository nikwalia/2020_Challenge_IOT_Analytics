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
    for channel in df.columns:
        df[channel] = gaussian_filter1d(df[channel].to_numpy(), 5)
    
    first_quartiles = df.quantile(q = 0.25, axis = 0)
    third_quartiles = df.quantile(q = 0.75, axis = 0)
    iqr = third_quartiles - first_quartiles

    for channel in df.columns:
        df[channel].loc[df[channel] > third_quartiles[channel] + 1.5 * iqr[channel]] = np.NAN
        df[channel].loc[df[channel] < first_quartiles[channel] - 1.5 * iqr[channel]] = np.NAN

    df_interpol = df_interpol.interpolate().replace(np.NAN, 0)
    return df_interpol

def get_outliers(df):
    first_quartiles = df.quantiles(q = 0.25)
    third_quartiles = df.quantiles(q = 0.75)
    iqr = third_quartiles - first_quartiles
    outliers = df.loc[(df < first_quartiles - 1.5 * iqr) & (df > third_quartiles + 1.5 * iqr)]
    return outliers


def get_stats_list(df):
    return [calc_mean(df),calc_median(df),calc_std(df),calc_max(df),calc_min(df)]
