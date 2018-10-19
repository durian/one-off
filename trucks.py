import pandas as pd
import numpy as np
from collections import Counter
import sys
import math
import argparse
import os

'''
Creates the delta's between the histograms on a per truck basis.

All data is saved in delta.csv, containing trucks histogram deltas
in chronological order per truck.

The delta.csv contains the following fields:
  VEHICL_ID;T_CHASSIS;PARAMETER_CODE;truck_date;1_1;...;20_20;valid;repaired;delta_days;delta_cells

TODO: add a normalisation parameter (per day?)

'''

parser = argparse.ArgumentParser()
parser.add_argument( '-t', "--truck", type=str, default=None, help='Truck ID' )
parser.add_argument( '-o', "--output", type=str, default="delta.csv", help='Output file' )
parser.add_argument( '-n', "--normalise", action='store_true', default=False, help='Normalise' )
parser.add_argument( "--top", type=int, default=None, help='Top n' )
parser.add_argument( "--rows", type=int, default=None, help='Rows to read' )
parser.add_argument( "--min_hists", type=int, default=0, help='Truck should at least have this many histograms' )
parser.add_argument( "--skip_zeroes", action='store_true', default=False, help='Skip if zero change' )
parser.add_argument( "--skip_resets", action='store_true', default=False, help='Skip truck if data reset' )
parser.add_argument( "--skip_nans", action='store_true', default=False, help='Skip truck if NaNs in data' )
args = parser.parse_args()

# ['VEHICL_ID', 'T_CHASSIS', 'PARAMETER_CODE', 'truck_date', '1_1', ..., '20_20', 'valid', 'repaired']
#           0            1                 2             3      4           403      404         405
print( "Reading data...", end="", flush=True)
lines = []#list(range(1, 100000))
df_train = pd.read_csv( "data_frame--2018-09-25--11-17-39.csv",
                        sep=";", dtype={'VEHICL_ID': str},
                        #https://stackoverflow.com/questions/17465045/can-pandas-automatically-recognize-dates
                        parse_dates=['truck_date'],
                        #skiprows=lines,
                        nrows=args.rows
)
print( "Ready" )
print( df_train.shape )
print( df_train.head(2) )
#print( list(df_train.columns) )

# VEHICL_ID;T_CHASSIS;PARAMETER_CODE;truck_date;....;valid;repaired
the_label      = "repaired"
the_id         = "T_CHASSIS" 
the_date       = "truck_date"

print( "Repaired\n", df_train[the_label].value_counts() )
print( df_train[(df_train[the_label]==1)] )

if args.truck:
    the_list = [ args.truck ]
else:
    the_list = df_train[the_id].value_counts().index.tolist()
    if args.top:
        the_list = the_list[0:args.top]
    #print( df_train[the_id].value_counts() )

delta_cols = list(df_train.columns) + [ "delta_days", "delta_cells" ]
#print( delta_cols )
with open(args.output, "w") as f:
    cols_str = ";".join(delta_cols)
    f.write( "{}\n".format(cols_str) )

#delta_df = pd.DataFrame( columns=delta_cols ) # now direct write to delta.txt

histogram_count = 0
repairs_count   = 0
negatives_count = 0
nans_count      = 0
trucks_count    = 0
for  the_vehicle_id in the_list:
    try:
        df_train1 = df_train[ ( df_train[the_id] == the_vehicle_id ) ]
    except:
        print( "ERROR..." )
        continue
    
    df_train1 = df_train1.sort_values(by=[the_date])
    num_hists = df_train1.shape[0]
    
    print( the_vehicle_id, trucks_count, num_hists, histogram_count )
    histogram_count += num_hists

    if num_hists < args.min_hists:
        print( "Too few histograms", the_vehicle_id )
        continue
    
    hist_data    = df_train1.iloc[ :,    4:404]
    hist_pre     = df_train1.iloc[ :,    0:  4]
    hist_post    = df_train1.iloc[ :,  404:406]
    #
    hist_labels  = df_train1.loc[ :, the_label]
    hist_chassis = df_train1.loc[ :, the_id]
    hist_sdate   = df_train1.loc[ :, the_date]
    hist_part    = "" #df_train1.loc[ :, "PARTITIONNING"]
    #print( hist_data.head(2) )
    #print( hist_sdate.head(2) )
    #print( hist_sdate )
    #
    ## subtractvector = first vehicle data #then subtract this from hist, and update for next
    #
    subtract_v   = hist_data.iloc[0,:]
    timestamp_p  = hist_sdate.iloc[0]
    truck_hists  = [] #collect them first
    negatives    = False
    repairs      = False
    nans         = False
    for idx in range(0, num_hists):
        hist       = hist_data.iloc[idx,:]  
        hist_delta = hist - subtract_v
        
        used_hours = hist_delta.sum() #should give total hours used in this period?
        hist_label = hist_labels.iloc[idx] #idx was i
        timestamp  = hist_sdate.iloc[idx]
        delta_mins = int((timestamp - timestamp_p).total_seconds()/60.0)  #minutes
        #delta_days = (str(timestamp - timestamp_p).split())[0]
        # delta_days was a bit tricky, as we want it on a date basis
        delta_days = np.busday_count(timestamp_p, timestamp, weekmask=[1,1,1,1,1,1,1])

        # Number of cells with changed value
        delta_cells = np.count_nonzero(hist_delta)
        
        # Short ones:
        #  Zero 2017-12-14 11:58:41 2017-12-14 11:58:41
        #  Zero 2018-02-04 10:17:43 2018-02-04 11:29:35
        #  Zero 2018-03-14 07:35:25 2018-03-14 10:44:51
        if args.skip_zeroes and idx > 0:
            if delta_days == 0 or delta_cells == 0:
                #print( "Zero", timestamp_p, timestamp, delta_cells )
                # Skip it ?
                continue
        if hist_label == 1:
            #print( "Repair", timestamp_p, timestamp, delta_cells )
            repairs = True

        # Check for negatives/resets? what to do?
        if (hist_delta<0).any():
            #print( "Negatives", timestamp_p, timestamp, delta_cells )
            negatives = True
            #sys.exit(1)
            #continue #doesn't make sense, next one will also be negative after reset
            # skip whole truck? skip rest of data?
        
        #timestamp  = timestamp[0:10] # lose the time

        if args.normalise and delta_days > 0:
            hist_delta = hist_delta / delta_days #used_hours #normalise over days/hours used

        if np.isnan(hist_delta).any():
            nans = True
            
        q = list(hist_pre.iloc[idx,:]) + list(hist_delta) + list(hist_post.iloc[idx,:]) + [delta_days, delta_cells]
        q_str = [str(x) for x in q]
        truck_hists.append( q_str )
        '''
        with open(args.output, "a") as f:
            cols_str = ";".join(q_str)
            f.write( "{}\n".format(cols_str) )
        '''
        timestamp_p = timestamp
        subtract_v  = hist
    # save truck_hist in one go
    if num_hists < args.min_hists: #check again
        print( "Too few histograms", the_vehicle_id )
        continue
    if negatives:
        negatives_count += 1
        print( "--> Negatives", negatives_count )
        if args.skip_resets:
            continue
    if nans:
        nans_count += 1
        print( "--> NaNs", nans_count )
        if args.skip_nans:
            continue
    if repairs:
        repairs_count += 1 # we count trucks with repairs
        print( "--> Repairs", repairs_count )
    #
    trucks_count += 1
    with open(args.output, "a") as f:
        for q_str in truck_hists:
            cols_str = ";".join(q_str)
            f.write( "{}\n".format(cols_str) )
print( "Trucks {} Repairs {} Negatives {} NaNs {}".format(trucks_count, repairs_count, negatives_count, nans_count) )

