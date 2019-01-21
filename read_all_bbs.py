import pandas as pd
import pytz
import matplotlib as mpl
mpl.use("Pdf")
import matplotlib.pyplot as plt
import numpy as np
import glob, os, subprocess
import argparse
import datetime
import time

'''
Differences... 240 frames = 10 seconds?

1530937013599
Assuming that this timestamp is in milliseconds:
GMT: Saturday, 7 July 2018 04:16:53.599
Your time zone: Saturday, 7 July 2018 06:16:53.599 GMT+02:00 DST
Relative: 6 months ago


1530881813617
Assuming that this timestamp is in milliseconds:
GMT: Friday, 6 July 2018 12:56:53.617
Your time zone: Friday, 6 July 2018 14:56:53.617 GMT+02:00 DST
Relative: 6 months ago

'''

parser = argparse.ArgumentParser()
parser.add_argument( '-f', "--filename", type=str, default="FOO.csv", help='CSV filename' )
parser.add_argument( '-r', "--results", type=str, default="./results", help='Output files (like results_camera-80)' )
parser.add_argument( '-n', "--no_save", action='store_true', default=False, help="Don't save CSV file" )
args = parser.parse_args()

def custom_round(x, base=5):
    return int(base * round(float(x)/base))

folder = args.results #"./results" #folder = "results_camera-80"
filenames = glob.glob(os.path.join(folder, "Experiment*/*bboxcords.txt"))
print( filenames )
all_bbs = []
processed = [] # we can have doubles
# problems with "na" in data, &c
colours = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
           "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
           "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
           "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"] # add a colour, not necessary but convenient
colours = ["#1f77b4", "#aec7e8"] # add a colour, not necessary but convenient
colour_idx = 0
processed_info = []
for filename in sorted(filenames):
    filename_base = os.path.basename(filename)
    #tl_20180706T121648Z-1530937013599
    #live5-1546902576579.ts_bboxcords.txt
    f_type = "timelapse"
    if filename_base[0:2] == "tl":
        filename_date = filename_base[3:7]+"/"+filename_base[7:9]+"/"+filename_base[9:11] #filename_base[3:11]
        filename_time = filename_base[12:14]+":"+filename_base[14:16]+":"+filename_base[16:18] #filename_base[12:18]
        epoch_raw = "r"+filename_base[20:33]
        epoch_val = int(float(filename_base[20:33])/1000)
        film_date = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
    else:
        f_type        = "live"
        filename_date = "??" # we don't really care about these
        filename_time = "??"
        epoch_val     = int(float(filename_base[6:19])/1000) #on the movie is local time
        film_date = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
        epoch_raw = "r"+str(epoch_val)
    print( filename_base, epoch_raw, film_date, film_time ) # we'll get some extra conversions
    # We have doubles sometimes, skip them.
    if epoch_raw in processed:
        continue
    processed.append( epoch_raw )
    #
    epoch_str = datetime.datetime.utcfromtimestamp(epoch_val) 
    #print( filename_date, filename_time, epoch_str )
    with open(filename, "r") as f:
        lines = (f.readlines())[1:]
        for line in lines:
            epoch_str = datetime.datetime.utcfromtimestamp(epoch_val)
            bits = line.strip().split(",") + [filename_base, epoch_val, epoch_str]
            if " na" not in bits:
                # calculate area? diff from previous, etc?
                l = float(bits[5]) - float(bits[3]) # length, xmax - xmin
                h = float(bits[6]) - float(bits[4]) # height, ymax - ymin
                a = l * h                           # area
                cx = float(bits[3]) + l/2.0         # centre
                cy = float(bits[4]) + h/2.0         # centre
                bits += [a, cx, cy]
                bits.append( colours[colour_idx] )
                all_bbs.append( bits )
            else:
                bits = [bits[0], 0, 0, 0, 0, 0, 0, filename_base, epoch_val, epoch_str, 0, 0, 0, colours[colour_idx]] 
                all_bbs.append( bits )
            if f_type == "live":
                epoch_val += 1/5 #  5 fps
            else:
                epoch_val += 5 # one every 5 seconds # 1/24.0 # add one frame time
    colour_idx += 1
    colour_idx = colour_idx % len(colours)
    end_date = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
    end_time = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
    processed_info.append( [filename_base, epoch_raw, film_date, film_time, end_date, end_time] )

#print( all_bbs )
df = pd.DataFrame( all_bbs,
                   columns=["framenum", "class", "conf", "xmin", "ymin", "xmax", "ymax", "file", "raw", "epoch", "area", "cx", "cy", "colour"]
)

df.sort_values(['raw', 'framenum'], ascending=[True, True], inplace=True)
#df = df.reset_index(drop=True)

print( df.info() )
print()
print( df.describe() )
print()
print( df.head(10) )
print()
print( df.tail(10) )

print( df["file"].unique() )
print( df["epoch"].unique() )

if not args.no_save:
    print( "Saving", args.filename )
    df.to_csv( args.filename, sep=";", index=False )

for info in processed_info:
    print( info )
