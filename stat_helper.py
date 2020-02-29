import pandas as pd

def calc_std(dataframe):
    return dataframe.std(axis = 1)

def calc_mean(dataframe):
    return dataframe.mean(axis = 1)
