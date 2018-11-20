import numpy as np
import random
import sys
import pandas as pd
from datetime import datetime, timedelta
from Representation import *
from ConformalAnomaly import *
from Interestingness import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_csv("df.csv", header=None, sep=";")
data[0] = pd.to_datetime(data[0], format='%Y-%m-%d %H:%M:%S')
#print( data.head() )
df = pd.DataFrame(data[1].values, index=data[0]) #create timeseries
print( df.head(4) )
print( df.tail(4) )
print( df.shape )


fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
c = '#4A7BC9'
plot_type = "vlines"
if plot_type == "vlines":
    ax0.vlines(x=df.index,
               ymin=0,
               #ymax=cam["average_movement"].values, color=c, alpha=0.5,
               ymax=df.values, color=c, alpha=0.5,
               label="average_movement")
else:
    ax0.bar( df.index, df.values,
             width=1/25, #in "days", 1/24 is hour width
             align="edge",
             color=c,
             #edgecolor="black",
    )
#
#cam.plot( x="dt",
#            y="average_movement" )
plt.gcf().autofmt_xdate()
fig0.autofmt_xdate()
#https://stackoverflow.com/questions/12945971/pandas-timeseries-plot-setting-x-axis-major-and-minor-ticks-and-labels
myFmt = mdates.DateFormatter('%m-%d %H:%M')
#plt.gca().xaxis.set_major_formatter(myFmt)
ax0.xaxis.set_major_formatter(myFmt)




type_rep   = 'moments' # 'moments', 'histogram', 'pairwise', or 'all'
w_rep      = '1d' # window to compute a representation
w_rg       = '7d' # window to take data from reference group
w_dl       = 30 # window to compute deviation level
uniformity = 'martingale' # 'kstest' or 'martingale' or 'martingale_multiplicative'
ncm        = 'median' # 'median' or 'knn'
k          = 10 # used when the non-conformity measure (ncm) is knn

rep = Representation( type_rep )
dfs = rep.extract_all_units( [ [df] ] )
print( dfs )

try:
        iness = Interestingness(dfs)
        iness_scores = [ iness.clustering_quality(k) for k in [2, 4] ] + [ iness.dispersion() ]
        print("interestingness_scores = ", iness_scores)
except ValueError:
        pass

cfa = ConformalAnomaly( w_dl, uniformity, ncm )
print( cfa.cosmo( dfs, w_rg, w_rep ) )

#dfs1 = rep.extract_all_units( [ [df1] ] )
#print( cfa.cosmo2( dfs1, dfs, w_rg, w_rep ) )

plt.show(block=True)
