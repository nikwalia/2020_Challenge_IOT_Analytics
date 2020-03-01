import hdf_helper
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.sparse import csc_matrix
import os

df_ch_1 = pd.read_csv('dat/dat_ch_1.csv').drop(['Unnamed: 0'], axis = 1)
df_ch_1 = df_ch_1.replace(np.NAN, 0)
df_csc = csc_matrix(df_ch_1.to_numpy())

clf = IsolationForest().fit(df_ch_1)
for i in np.arange(df_ch_1.shape[1]):
    print(i)
    sns.distplot(df_ch_1[:, i])
plt.show()