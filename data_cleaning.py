import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, RobustScaler

'''
Perform a 1-dimensional Gaussian Filter on the data to smooth out small outliers.

return smoothed values
'''
def smooth_values(df):
    filtered = pd.DataFrame()
    for channel in df.columns:
        filtered[channel] = gaussian_filter1d(df[channel].to_numpy(), 1)
    
    return filtered

'''
Scales all values in a DF between 0 and 1
ONLY USE ON A SMOOTHED DATASET

return scaled dataframe
'''
def min_max_scaling(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    return scaler.fit(df)

'''
Scales all values in a DF between 0 and 1 with IQR to account for outliers

return scaled dataframe
'''
def robust_scaling(df):
    scaler = RobustScaler()
    return scaler.fit(df)

'''
Gets a DF that contains the averages of every n elements (cluster_size) to reduce size of data

return df of averages
'''
def reduce_dataset_size(df, cluster_size=100):
    df = df.rolling(cluster_size).mean()
    df = df.iloc[::cluster_size, :].dropna()
    return df