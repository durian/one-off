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
Date/Time info from timelapse_new folder
'''

parser = argparse.ArgumentParser()
parser.add_argument( '-f', "--folder", type=str, default="timelapse_new", help='Files' )
args = parser.parse_args()

folder = args.folder
filenames = glob.glob(os.path.join(folder, "Camera-*/*"))
processed = []
processed_info = []
for filename in sorted(filenames):
    filename_dir = os.path.basename(os.path.dirname( filename )) # gets last directory before filename
    filename_base = os.path.basename(filename)
    #tl_20180706T121648Z-1530937013599
    #live5-1546902576579.ts_bboxcords.txt
    f_type = "timelapse"
    if filename_base[0:2] == "tl":
        epoch_raw     = "r"+filename_base[20:33]
        epoch_val     = int(float(filename_base[20:33])/1000)
        epoch_start   = epoch_val
        film_date     = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time     = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
    elif filename_base[0:5] == "live5":
        f_type        = "live"
        epoch_val     = int(float(filename_base[6:19])/1000) #on the movie is local time
        epoch_start   = epoch_val
        film_date     = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time     = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
        epoch_raw     = "r"+str(epoch_val)
    elif filename_base[0:6] == "live30":
        f_type        = "live"
        epoch_val     = int(float(filename_base[7:20])/1000) #on the movie is local time
        epoch_start   = epoch_val
        film_date     = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time     = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
        epoch_raw     = "r"+str(epoch_val)
    else:
        print( "ERROR, unknow format." )
        sys.exit(1)
    epoch_str = datetime.datetime.utcfromtimestamp(epoch_val) 
    #end_date  = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
    #end_time  = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
    processed_info.append( [filename_dir, filename_base, epoch_start, film_date, film_time] )

for info in processed_info:
    print( info )
