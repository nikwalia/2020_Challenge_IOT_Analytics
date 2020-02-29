import pandas as pd

def calc_std(df):
    return df.std(axis = 0)

def calc_mean(df):
    return df.mean(axis = 0)

def calc_max(df):
    return df.max(axis = 0)

def calc_min(df):
    return df.min(axis = 0)

def calc_median(df):
    return df.median(axis = 0)

def remove_outliers(df):
    first_quartiles = df.quantile(q = 0.25, axis = 0)
    third_quartiles = df.quantile(q = 0.75, axis = 0)
    iqr = third_quartiles - first_quartiles
    non_outliers = df.loc[(df >= first_quartiles - 1.5 * iqr) & (df <= third_quartiles + 1.5 * iqr)]
    return non_outliers