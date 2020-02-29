import pandas as pd

def calc_std(df):
    return df.std(axis = 1)

def calc_mean(df):
    return df.mean(axis = 1)

def calc_max(df):
    return df.max(axis = 1)

def calc_min(df):
    return df.min(axis = 1)

def calc_median(df):
    return df.median(axis = 1)

def find_outliers(df):
    first_quartiles = df.quantiles(q = 0.25, axis = 1)
    third_quartiles = df.quantiles(q = 0.75, axis = 1)
    iqr = third_quartiles - first_quartiles
    outliers = df.loc[df > third_quartiles + 1.5 * iqr | df < first_quartiles - 1.5 * iqr]
    print(outliers)