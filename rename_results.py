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
import shutil

'''
Rename the tl_20190115T103704Z-1547618230161.ts_bboxcords files in the
Experiment... folders.
'''

parser = argparse.ArgumentParser()
parser.add_argument( '-r', "--results", type=str, default="./results", help='Output files (like results_camera-80)' )
parser.add_argument( '-e', "--extra", type=str, default="CAM", help='Extra info, like cam80' )
args = parser.parse_args()

folder = args.results #"./results" #folder = "results_camera-80"
filenames = glob.glob(os.path.join(folder, "Experiment*/*bboxcords.txt"))
print( filenames )
all_bbs = []
processed = [] # we can have doubles
# problems with "na" in data, &c
for filename in sorted(filenames):
    filename_base = os.path.basename(filename)
    #tl_20180706T121648Z-1530937013599
    #live5-1546902576579.ts_bboxcords.txt
    f_type = "tl"
    if filename_base[0:2] == "tl":
        filename_date = filename_base[3:7]+"/"+filename_base[7:9]+"/"+filename_base[9:11] #filename_base[3:11]
        filename_time = filename_base[12:14]+":"+filename_base[14:16]+":"+filename_base[16:18] #filename_base[12:18]
        epoch_raw     = "r"+filename_base[20:33]
        epoch_val     = int(float(filename_base[20:33])/1000)
        epoch_start   = epoch_val
        film_date     = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time     = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
    elif filename_base[0:5] == "l5":
        f_type        = "live"
        filename_date = "??" # we don't really care about these
        filename_time = "??"
        epoch_val     = int(float(filename_base[6:19])/1000) #on the movie is local time
        epoch_start   = epoch_val
        film_date     = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time     = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
        epoch_raw     = "r"+str(epoch_val)
    elif filename_base[0:6] == "l3":
        f_type        = "live"
        filename_date = "??" # we don't really care about these
        filename_time = "??"
        epoch_val     = int(float(filename_base[7:20])/1000) #on the movie is local time
        epoch_start   = epoch_val
        film_date     = time.strftime('%Y/%m/%d',  time.gmtime(int(epoch_val)))
        film_time     = time.strftime('%H:%M:%S',  time.gmtime(int(epoch_val)))
        epoch_raw     = "r"+str(epoch_val)
    else:
        print( "ERROR, unknow format." )
        sys.exit(1)
    #print( filename_base, epoch_raw, film_date, film_time ) # we'll get some extra conversions
    # We have doubles sometimes, skip them.
    if epoch_raw in processed:
        continue
    processed.append( epoch_raw )
    #
    epoch_str = datetime.datetime.utcfromtimestamp(epoch_val)
    # Rename
    filename_dir = os.path.dirname( filename )
    new_filename_dir = os.path.split(filename_dir)[0]
    new_filename_base = args.extra + "_" + f_type+"_"+time.strftime('%Y%m%d_%H%M%S',  time.gmtime(int(epoch_val)))
    new_filename_base = new_filename_base + "_bbox.txt"
    new_filename = os.path.join(new_filename_dir, new_filename_base)
    print( filename_base, new_filename)
    shutil.copy(filename, new_filename)
