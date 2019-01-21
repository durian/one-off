#!/usr/bin/env python3
#
import re
import getopt, sys, os
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.dates as mdates
    
def custom_round(x, base=100):
    return int(base * round(float(x)/base))

parser = argparse.ArgumentParser()
parser.add_argument( '-f', "--filename", type=str, default="FOO.csv", help='CSV filename' )
parser.add_argument( '-t', "--type", type=str, default="vlines", help='Plot type' )
args = parser.parse_args()

df = pd.read_csv( args.filename, sep=";" )
df["ROUND"] = df.apply(lambda row: custom_round(row['area']), axis=1)
print( df )

# Loop per day/hour and produce multiple graphs? Or do this while reading data?
# Create a timeseries, select data between date/time boundaries.
start_date = df["epoch"].min()
end_date   = df["epoch"].max() #+ pd.Timedelta('1H')
dtr = pd.date_range( start=start_date, end=end_date, freq='1H').floor('H') #round("H")
print( start_date, "-", end_date, "\n", dtr )

fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
c = '#4A7BC9'
plot_type = args.type
if plot_type == "vlines":
    secs = mdates.epoch2num(df["raw"])
    ax0.vlines(secs, #x=df["epoch"], #x=df.index, #.values,
               ymin=0,
               #ymax=cam["average_movement"].values, color=c, alpha=0.5,
               ymax=df["ROUND"].values, color=df["colour"], alpha=0.5,
               label="area"
    )
    import matplotlib.dates as mdates
    plt.gcf().autofmt_xdate()
    fig0.autofmt_xdate()
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    #plt.gca().xaxis.set_major_formatter(myFmt)
    ax0.xaxis.set_major_formatter(myFmt)
elif plot_type == "test":
    delta_h = pd.Timedelta('1H') - pd.Timedelta('1s')
    num = len(dtr)
    empty = 0
    cmap = plt.cm.get_cmap('hsv', num)
    print( "Hours:", num )
    df["ts"] = pd.to_datetime(df["raw"], unit='s')
    for i,start_t in enumerate(dtr):
        #print( start_t, start_t + delta_h )
        t0 = start_t
        t1 = start_t+delta_h
        data = df[ (df["ts"] >= t0) & (df["ts"] < t1) ]
        if data.empty:
            empty += 1
        #print( data["epoch"] )
        secs = mdates.epoch2num(data["raw"])
        ax0.vlines(secs, 
                   ymin=0,
                   ymax=data["ROUND"].values,
                   color=cmap(i),
                   alpha=0.5,
                   label="area"
        )
    print( num, empty, num - empty )
    import matplotlib.dates as mdates
    plt.gcf().autofmt_xdate()
    fig0.autofmt_xdate()
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    #plt.gca().xaxis.set_major_formatter(myFmt)
    ax0.xaxis.set_major_formatter(myFmt)
elif plot_type == "centres":
    '''
    ax0 = df.plot.scatter(x='cx',
                y='cy',
                c=df['colour'])
    '''
    ax0.scatter(x=df['cx'],  # need to be normalised
                y=df['cy'],
                c=df['colour']
    )
elif plot_type == "hours": # "anchors" maybe when in stable, etc, not hours
    delta_h = pd.Timedelta('1H') - pd.Timedelta('1s')
    num = len(dtr)
    empty = 0
    cmap = plt.cm.get_cmap('hsv', num)
    print( "Hours:", num )
    for i,start_t in enumerate(dtr):
        #print( start_t, start_t + delta_h )
        df["ts"] = pd.to_datetime(df["raw"], unit='s')
        t0 = start_t
        t1 = start_t+delta_h
        data = df[ (df["ts"] >= t0) & (df["ts"] < t1) ]
        if data.empty:
            empty += 1
        #print( data["epoch"] )
        ax0.scatter(x=data['cx'],  # need to be normalised
                y=data['cy'],
                c=cmap(i)
    )
    print( num, empty, num - empty )
elif plot_type == "rectangles":
    pass
    # rect = plt.Rectangle((i, -0.5), 1, 1, facecolor=cmap(i))
    # ax.add_artist(rect)
else:
    ax0.bar( df.index, df["ROUND"],
             width=1, #/25, #in "days", 1/24 is hour width
             align="edge",
             color=df["colour"],
             #edgecolor="black",
    )
ax0.set_title( args.filename )

plt.show(block=True)
