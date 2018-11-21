#!/usr/bin/env python3
#
import re
import sys, os
import argparse
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
import matplotlib.dates as mdates

pd.set_option('display.width', 120)

# CameraMinuteSensorData.csv

parser = argparse.ArgumentParser()
parser.add_argument( '-f', '--filename', type=str, default="CameraMinuteSensorData.csv", help='File')
parser.add_argument( '-c', '--camid', type=int, default=53, help='Camera ID')
parser.add_argument( '-t', '--timeslice', type=str, default="300s", help='Resample time length')
parser.add_argument( '-d', '--data', type=str, default="exp_weighted_moving_average", help="Data column")
args = parser.parse_args()

df = pd.read_csv(args.filename)

df['dt'] = pd.to_datetime(df['timestamp'].astype(str), format='%Y-%m-%d %H:%M:%S')

print( df.head() )
print( df.tail() )

print( df["camera_unit_id"].value_counts() )

#id  camera_unit_id  average_movement  exp_weighted_moving_average            timestamp  load1  load5  load15
#0  5618446               8          0.000000                 4.619830e-14  2018-08-01 00:01:00   0.96   1.07    1.14

cam = df[(df["camera_unit_id"]==args.camid)]
cam = cam.set_index(["dt"])
#cam = cam[ (cam.index < "2018-08-07") ]
cam = cam.resample(args.timeslice).sum()
if cam.empty:
    print( "ERROR, selection is empty." )
    sys.exit(1)
print( cam.head() )

fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
c = '#4A7BC9'
plot_type = "vlines"
if plot_type == "vlines":
    ax0.vlines(x=cam.index.values,
               ymin=0,
               #ymax=cam["average_movement"].values, color=c, alpha=0.5,
               ymax=cam[args.data].values, color=c, alpha=0.5,
               label=args.data)
else:
    ax0.bar( cam.index, cam[args.data],
             width=1/25, #in "days", 1/24 is hour width
             align="edge",
             color=c,
             #edgecolor="black",
    )
ax0.set_title("camid "+str(args.camid)+" / "+args.data+" / "+args.timeslice)
#
#cam.plot( x="dt",
#            y="average_movement" )
plt.gcf().autofmt_xdate()
fig0.autofmt_xdate()
#https://stackoverflow.com/questions/12945971/pandas-timeseries-plot-setting-x-axis-major-and-minor-ticks-and-labels
myFmt = mdates.DateFormatter('%m-%d %H:%M')
#plt.gca().xaxis.set_major_formatter(myFmt)
ax0.xaxis.set_major_formatter(myFmt)
#
## https://matplotlib.org/api/dates_api.html#matplotlib.dates.HourLocator
##ax0.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
##ax0.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))

fn = "camid_{:03d}_{:s}_{:s}.png".format(args.camid, args.data, args.timeslice)
fig0.savefig(fn, dpi=300)
print( "Saved", fn )

plt.show(block=True)

df = cam[args.data]
print(df.head())
fn = "camid_{:03d}_{:s}_{:s}.csv".format(args.camid, args.data, args.timeslice)
df.to_csv(fn, sep=";", header=None)
print( "Saved", fn )
