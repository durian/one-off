import pandas as pd
import numpy as np
from collections import Counter
import sys
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
#from matplotlib.colors import LogNorm
from sklearn import preprocessing
import argparse
import os
import joblib

parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--anonymous", action='store_true', default=False, help='Hide axes' )
parser.add_argument( '-t', "--truck", type=str, default=None, help='Truck ID' )
parser.add_argument( '-f', "--filename", type=str, default="P1FW6_20181008.csv", help='CSV filename' )
parser.add_argument( '-n', "--normalise", action='store_true', default=False, help='Normalise' )
parser.add_argument( "--max_pics", type=int, default=16, help='Rows' )
parser.add_argument( "--cols", type=int, default=4, help='Cols' )
parser.add_argument( "--top", type=int, default=None, help='Top n' )
parser.add_argument( "--start", type=int, default=0, help='Start histogram' )
parser.add_argument( "--ncb", action='store_true', default=False, help='No colourbar' )
args = parser.parse_args()

print( "Reading data...", end="", flush=True)
lines = []#list(range(1, 100000))
df_train = pd.read_csv( args.filename,
                        sep=";", dtype={'VEHICLE_ID': str},
                        #skiprows=lines,
                        #nrows=10000
)
print( "Ready" )
print( df_train.shape )
print( df_train.head(2) )

# PARAMETER_CODE SEND_DATETIME T_CHASSIS  VEHICLE_ID  Y_INDEX_1_X_INDEX_1_VALUE ... Y_INDEX_10_X_INDEX_11_VALUE
# 0                  1             2          3          4                              113
the_label      = "PARAMETER_CODE" # dummy label
the_id         = "T_CHASSIS"
the_date       = "SEND_DATETIME"

uniqs = pd.unique( df_train[the_id] ) 
if args.truck:
    uniqs = [ args.truck ]
print( uniqs )

for the_vehicle_id in uniqs:
    ##
    print( the_vehicle_id )

    #df_train1 = df_train[ ( str(df_train[the_id]) == str(the_vehicle_id) ) ]
    try:
        df_train1 = df_train[ ( df_train[the_id] == the_vehicle_id ) ]
    except:
        print( "ERROR..." )
        continue
    
    df_train1 = df_train1.sort_values(by=[the_date])
    num_pics = df_train1.shape[0]
    if num_pics < 4:
        print( "skipping", the_vehicle_id, num_pics )
        continue

    # Plot the last ones, according to date
    if num_pics > args.max_pics:
        #df_train1 = df_train1.iloc[-args.max_pics:]
        #num_pics = df_train1.shape[0]
        st = num_pics - args.max_pics #st is the start in the array
        num_pics = args.max_pics

    #if args.start:
    st = args.start
 
    train_data    = df_train1.iloc[ :,  4:114]
    train_labels  = df_train1.loc[ :, the_label]
    train_chassis = df_train1.loc[ :, the_id]
    train_sdate   = df_train1.loc[ :, the_date]
    print( train_data.head(2) )
    print( train_sdate.head(2) )
    #
    sp = 1
    n  = 3 #4
    cols = args.cols
    rows = int(num_pics / cols) + 1*((num_pics % cols)!=0)
    nn = rows * cols
    pc = 0 # pic count
    #
    fig = plt.figure(figsize=(cols*3, rows*3)) #plt.figure(figsize=(sz,sz))
    plt.subplots_adjust( hspace=0.7 )
    #
    # Plot the last nn
    #if num_pics > nn:
    #    st = num_pics - nn
    #    print( "Starting at", st )
    #
    for i in range(st, st+nn):
        idx = i #ids.pop() # this plots same T_CHASSIS defined in triplets loop above
        try:
            im = train_data.iloc[idx,:] #idx was i
        except:
            ax.set_aspect('equal')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel( "" )
            ax.set_ylabel( "" )
            sp += 1
            continue
        ax = fig.add_subplot(rows,cols,sp) #was n, n, sp
        pc += 1
        hist_label = 0 #train_labels.iloc[idx] #idx was i
        timestamp  = train_sdate.iloc[idx] 
        timestamp  = timestamp[0:10] # lose the time
        partition  = "" #train_part.iloc[idx][0]
        #qprint( i, hist_label )
        if hist_label == 0:
            ax.set_title( str(i)+"/l="+str(hist_label)+" "+str(timestamp)+" "+partition )
        else:
            ax.set_title( str(i)+"/l="+str(hist_label)+" "+str(timestamp)+" "+partition, color="red" )
        #im = minmax_scale(im)
        #im = im / np.linalg.norm(im)
        im = np.reshape( im, (11, 10) )
        im = np.rot90(im) #flipud(im) #rot90()
        #print( im )
        #
        if args.normalise:
            _min = 0
            _max = 1
            im += -(np.min(im))
            im /= np.max(im) / (_max - _min)
            #im += _min
        #
        im_masked = np.ma.masked_where(im == 0, im)
        plt.imshow( im_masked, interpolation='none')# vmin=0, vmax=10)
        ax.set_aspect('equal')
        ax.get_xaxis().set_ticks([0, 10])
        ax.get_yaxis().set_ticks([])
        #
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
        ax.set_yticks(np.arange(-.5, 9, 1), minor=True);
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        #
        #ax.set_ylim(ax.get_ylim()[::-1]) #upside down. but all data
        if not args.anonymous:
            ax.set_xlabel( "" )
            ax.set_ylabel( "" )
            if not args.ncb:
                plt.colorbar(orientation='vertical', ax=ax, format='%.1f', fraction=0.046, pad=0.04)
        else:
            if not args.ncb:
                plt.colorbar(orientation='vertical', ax=ax, ticks=[])
        sp += 1
    if not args.anonymous:
        fig.suptitle( the_id+": "+str(the_vehicle_id)+' P1FW6' ) #Training data '+str(st)+" +"+str(nn) )
    fn = "P1FW6_"+str(the_vehicle_id)+"_"+str(st)+"+"+str(pc)
    if args.normalise:
        fn += "_N"
    fn += ".png"
    print( "Saving", fn )
    fig.savefig( fn, dpi=288, transparent=True )
    #plt.show()
