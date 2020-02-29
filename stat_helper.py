import pandas as pd

def calc_mean(df):
    return df.mean()

def calc_max(df):
    return df.max()

def calc_min(df):
    return df.min()

def calc_median(df):
    return df.median()

def find_outliers(df):
    first_quartiles = df.quantiles(q = 0.25)
    third_quartiles = df.quantiles(q = 0.75)
    iqr = third_quartiles - first_quartiles
    non_outliers = df.loc[(df >= first_quartiles - 1.5 * iqr) & (df <= third_quartiles + 1.5 * iqr)]
    return non_outliers