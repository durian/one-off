import re
import sys, os
import argparse
from datetime import datetime
from datetime import timedelta
#https://stackoverflow.com/questions/35066588/is-there-a-simple-way-to-increment-a-datetime-object-one-month-in-python
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools

from read_data import *
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


#my_ts = pd.read_csv( "3dprodmonths_2010-01_fgrp3111_120m.csv", sep=";", index_col=0 )
my_ts = pd.read_csv( "3dprodmonths_2012-01_fgrp3111_48m.csv", sep=";", index_col=0 )
print( my_ts )
X = my_ts.values[:,0:25]
print( X )

seed = 0
num = 2
ks = KShape(n_clusters=num, verbose=True, random_state=seed)
y_pred = ks.fit_predict(X)
sz = X.shape[1]

plt.figure()
for yi in range(num):
    plt.subplot(num, 1, 1 + yi)
    for xx in X[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.show()
