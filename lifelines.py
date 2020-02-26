#!/usr/bin/env python3
#
import re
import random
import sys, os
import argparse
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.width', 120)

# ----

def gen_data():
    l = []
    # ["ID", "KM", "DEAD", "ENGINE", "MOUNTAIN", "CITY", "MONDAY"]
    for x in range(0, 1000):
        E  = random.randint(0,   1)
        M  = random.randint(0,   1)
        C  = random.randint(0,   1)
        MM = random.randint(0,  10)
        D  = random.randint(0,   4)
        if D > 0:
            D = 1
        if E == 0:
            km = 10000 + random.gauss(40000, 10000) # engine 0 operates less
        else:
            km = 10000 + random.gauss(60000, 10000) # than engine 1
        if M == 1:
            km -= random.gauss(8000, 1500) # mountain usage decreases
        if C == 1:
            km -= random.gauss(5000, 1500) # city usage decreases (less than mountain)
        if MM == 1:
            km = 100 + random.gauss(10000, 5000) # Monday morning model, just low KM and always breaks
            D  = 1
        else:
            #MM = 0
            pass
        if km > 0:
            d = [x, int(km), D, E, M, C, MM]
            l.append( d )
    return l

dl = gen_data()
df = pd.DataFrame( dl, columns=["ID", "KM", "DEAD", "ENGINE", "MOUNTAIN", "CITY", "MONDAY"] )
print( df )
#sys.exit(1)


from lifelines import KaplanMeierFitter, WeibullFitter

print(df.head())

EN = ( (df['ENGINE']   == 1), "engine" )
MO = ( (df['MOUNTAIN'] == 1), "mountain" )
CI = ( (df['CITY']     == 1), "city" )
MM = ( (df['MONDAY']   == 1), "Monday" )

BROKEN = ( (df['DEAD']  > 0),  "all broken down" )
OKAY   = ( (df['DEAD'] == 0), "all okay" ) # not used obviously

fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))

T  = df['KM']
E  = df['DEAD'] # all broken trucks
wf = WeibullFitter().fit(T, E)
wf.print_summary()
wf.plot_survival_function(ax=ax0, ci_show=False, label="Full population")

kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E) 

kmf.survival_function_
kmf.cumulative_density_

kmf.plot_survival_function(ax=ax0, ci_show=False)
#kmf.plot_cumulative_density()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
for X, x in [BROKEN, EN, MO, CI, MM]:
    T  = df['KM'][X]
    E  = df['DEAD'][X]

    print( x ) 

    wf = WeibullFitter().fit(T, E)
    wf.print_summary()
    #wf.plot(ax=ax0)
    #wf.plot_survival_function(ax=ax)
    
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label=x) 

    #kmf.survival_function_
    #kmf.cumulative_density_

    kmf.plot_survival_function(ax=ax, ci_show=False) 
    #kmf.plot_cumulative_density()

plt.show(block=True)

print( "CoxPHFitter()" )

from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='KM',  event_col='DEAD')
cph.print_summary()

estimation = cph.baseline_survival_
print( estimation )
 
hazard = cph.baseline_cumulative_hazard_
print(cph.score_)
print(cph.summary)

cph.plot()

censored_subjects = df.loc[df['DEAD'] == 0]
num_cs = len(censored_subjects)
print( num_cs )
# Add tailor made truck driven nnn km
d = [[1000, 10000, 0,  0, 0, 0, 2],
     [1000, 10000, 0,  1, 0, 0, 2],
     [1000, 10000, 0,  0, 1, 0, 2],
     [1000, 10000, 0,  1, 1, 0, 2]]
num_d = len(d) 

dfn = pd.DataFrame( d, columns=["ID", "KM", "DEAD", "ENGINE", "MOUNTAIN", "CITY", "MONDAY"] )
print( dfn )
censored_subjects = censored_subjects.append( dfn, ignore_index=True )

print( censored_subjects )

unconditioned_sf = cph.predict_survival_function(censored_subjects)
print( unconditioned_sf )

from lifelines.utils import median_survival_times, qth_survival_times
predictions_75 = qth_survival_times( 0.75, unconditioned_sf )
predictions_25 = qth_survival_times( 0.25, unconditioned_sf )
predictions_50 = median_survival_times( unconditioned_sf )
print( predictions_50 )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
for f in unconditioned_sf:
    ax.plot( unconditioned_sf[f], alpha=.5, label=f )
#ax.legend()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
for i,f in enumerate(reversed(unconditioned_sf.columns)):
    #print( i )
    if i < num_d:
        print( i, f )
        ax.plot( unconditioned_sf[f], alpha=1, label=f )
    else:
        ax.plot( unconditioned_sf[f], alpha=0.1, label=f, c='grey' )

plt.show(block=True)



