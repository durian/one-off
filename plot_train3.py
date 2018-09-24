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


parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--anonymous", action='store_true', default=False, help='Hide axes' )
parser.add_argument( '-n', "--normalise", action='store_true', default=False, help='Normalise' )
parser.add_argument( "--max_pics", type=int, default=16, help='Rows' )
parser.add_argument( "--cols", type=int, default=4, help='Cols' )
args = parser.parse_args()

#df = pd.read_csv( "mirrored.csv", sep=";" )

# T_CHASSIS;VEHICLE_ID;PARAMETER_CODE;Y_INDEX_1_X_INDEX_1_VALUE;Y_INDEX_2_X_INDEX_1_VALUE;...
df = pd.read_csv( "multi_flatten_v3_20100101-20190101_0.csv", sep=";", dtype={'VEHICLE_ID': str} )
df_train   = df #df[ ( df["PARTITIONNING"] == "1_Training" ) ]

#print( df_train["T_CHASSIS"].value_counts() )
'''
"B-699946" -> nice plot!
'''
the_label      = "All_Fault_in_3_months"
the_id         = "T_CHASSIS" #"VEHICLE_ID"

uniqs = pd.unique( df_train[the_id] ) 
print( uniqs )

#normalise = False #True
#max_pics  = 36

#for the_vehicle_id in uniqs:
for  the_vehicle_id in ["A-788914", "B-697108", "B-699946", "A-810435", "A-768393", "B-762810", "A-774841", "A-784917", "B-784203", "B-781780", "A-776912", "A-786632", "A-761610", "A-797571", "B-726445", "B-738451"]:
    #the_vehicle_id = "A-768393" #"B-699946" #"B-697108" #"A-800553" 
    ##
    print( the_vehicle_id )

    try:
        df_train1 = df_train[ ( df_train[the_id] == the_vehicle_id ) ]
        df_train1 = df_train1.sort_values(by=['Send_Date'])
    except:
        continue

    num_pics = df_train1.shape[0]
    if num_pics < 8:
        print( "skipping", the_vehicle_id, num_pics )
        continue

    # Plot the last ones, according to date
    if num_pics > args.max_pics:
        df_train1 = df_train1.iloc[-args.max_pics:]
        num_pics = df_train1.shape[0]
    
    train_data    = df_train1.iloc[ :,  3:403] 
    train_labels  = df_train1.loc[ :, the_label]
    train_chassis = df_train1.loc[ :, the_id]
    train_sdate   = df_train1.loc[ :, "Send_Date"]
    train_part    = df_train1.loc[ :, "PARTITIONNING"]
    print( train_sdate )
    #
    sp = 1
    st = 00
    n  = 3 #4
    cols = args.cols
    rows = int(num_pics / cols) + 1*((num_pics % cols)!=0)
    #nn = n * n
    nn = rows * cols
    #sz = n*3
    pc = 0 # pic count
    #
    fig = plt.figure(figsize=(cols*3, rows*3)) #plt.figure(figsize=(sz,sz))
    plt.subplots_adjust( hspace=0.7 )
    #
    # Plot the last nn
    if num_pics > nn:
        st = num_pics - nn
        print( "Starting at", st )
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
        hist_label = train_labels.iloc[idx] #idx was i
        chassis_id = train_sdate.iloc[idx] #train_chassis.iloc[idx] #THIS IS DATE NOW
        partition  = train_part.iloc[idx][0]
        #qprint( i, hist_label )
        if hist_label == 0:
            ax.set_title( str(i)+"/l="+str(hist_label)+" "+str(chassis_id)+" "+partition )
        else:
            ax.set_title( str(i)+"/l="+str(hist_label)+" "+str(chassis_id)+" "+partition, color="red" )
        #im = minmax_scale(im)
        #im = im / np.linalg.norm(im)
        im = np.reshape( im, (20, 20) )
        im = np.flipud(im) #rot90()
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
        ax.get_xaxis().set_ticks([0, 19])
        ax.get_yaxis().set_ticks([])
        if not args.anonymous:
            ax.set_xlabel( "engine speed" )
            ax.set_ylabel( "engine torque" )
            plt.colorbar(orientation='vertical', ax=ax)
        else:
            plt.colorbar(orientation='vertical', ax=ax, ticks=[])
        sp += 1
    if not args.anonymous:
        fig.suptitle( the_id+": "+str(the_vehicle_id)+' P1FWM' ) #Training data '+str(st)+" +"+str(nn) )
    fn = "training_"+str(the_vehicle_id)+"_"+str(st)+"+"+str(pc)
    if args.normalise:
        fn += "_N"
    fn += ".png"
    print( "Saving", fn )
    fig.savefig( fn, dpi=288 )
    #plt.show()
