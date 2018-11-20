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

date_today = datetime.now()
dfl = []
for x in range(0,10):
        days = pd.date_range(date_today, date_today + timedelta(28), freq='1H')
        data = np.random.random(size=len(days))

        if x in [2,3]:
                s = int(len(data)/2)
                data[s:s+30] = data[s:s+30] * 1 +0.5  #anomaly
        df = pd.DataFrame(data, index=days)
        dfl.append( df )
print( dfl[2].shape )
print( dfl[2].tail() )
print( dfl[0].tail() )

for df in dfl:
        df.plot()
plt.show(block=True)

data1 = np.random.random(size=len(days))
df1   = pd.DataFrame(data1, index=days)

type_rep   = 'moments' # 'moments', 'histogram', 'pairwise', or 'all'
w_rep      = '1d' # window to compute a representation
w_rg       = '1d' # window to take data from reference group
w_dl       = 30 # window to compute deviation level
uniformity = 'martingale' # 'kstest' or 'martingale' or 'martingale_multiplicative'
ncm        = 'median' # 'median' or 'knn'
k          = 10 # used when the non-conformity measure (ncm) is knn

rep = Representation( type_rep )
dfs = rep.extract_all_units( [ dfl ] )
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
