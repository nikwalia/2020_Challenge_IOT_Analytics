from stat_helper import *
from hdf_helper import *

import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
import h5py

from scipy.stats import pearsonr,spearmanr

pearson_trshld = 0.95
spearman_trshld = 0.95


'''
Finds the correlation between two time series across 3 different metrics
'''
def get_correlation(data1, data2):
    # Linear Correlation
    pearson_corr, _ = pearsonr(data1, data2)

    # Multivariate Correlation
    spearman_corr, _ = spearmanr(data1, data2)
    
    # Covaraiance
    covariance = np.cov(data1, data2)
    
    return pearson_corr, spearman_corr, covariance


'''
From a dataframe of time series, find all of the channel pairs that have
a covariance higher than the threshold
'''
def get_related_channels(df):
    related_channels = []
    for col in df.columns:
        for subcol in df.columns:
            if col != subcol:
                pearson, spearman, covariance = get_correlation(df[col],df[subcol])
                
                if (abs(pearson) > pearson_trshld and abs(spearman) > spearman_trshld):
                    related_channels.append([col,subcol])
                    # print("Channels: ", col, subcol)
                    # print(pearson, spearman)
                    # print("Correlation: ", covariance[1][0], '\n')

    return related_channels
                