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
def down_sample(x, f=50):
    # pad to a multiple of f, so we can reshape
    # use nan for padding, so we needn't worry about denominator in
    # last chunk
    xp = np.r_[x, np.NAN + np.zeros((-len(x) % f,))]
    # reshape, so each chunk gets its own row, and then take mean
    return np.nanmean(xp.reshape(-1, f), axis=-1)


'''
Helper function. Deals with dtype big_endian, found when reading in hdf files.
'''
def big_endian_problem(df):
    x = np.array(df, '>i4') # big endian
    newx = x.byteswap().newbyteorder()
    return pd.DataFrame(newx)