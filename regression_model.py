import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

neighbors = 20
# Get training data


# fit the model
clf = LocalOutlierFactor(n_neighbors=neighbors)
y_pred = clf.fit_predict(X)
y_pred_outliers = y_pred[200:]

# plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
