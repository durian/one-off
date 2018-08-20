import pandas as pd
import numpy as np
from collections import Counter
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import re
import glob
import argparse

parser = argparse.ArgumentParser()
# make thise lists, so we can have more
# parser.add_argument( '-f', '--fgrps', action='append', help='Specify FGRP4s')
parser.add_argument( '-u', "--hidden_units", type=str, action="append", help='Hidden units' )
parser.add_argument( '-d', "--dropout", type=float, action="append", help='Dropout probability' )
parser.add_argument( '-l', "--learning_rate", type=float, action="append", help='Learning rate' )
args = parser.parse_args()

print( args.dropout )

files = glob.glob("lENGINE_TYPE_*csv")

if args.hidden_units:
    keep = []
    for hu in args.hidden_units:
        subset = [ x for x in files if "_h"+hu in x]
        keep = keep + subset
    files = keep
    
if args.dropout:
    keep = []
    for do in args.dropout:
        if do == 0.0:
            subset = [ x for x in files if "_do" not in x] # dropout 0 has no _do in name
        else:
            subset = [ x for x in files if "_do"+str(do) in x]
        #print( subset )
        keep = keep + subset 
    files = keep

if args.learning_rate:
    keep = []
    for lr in args.learning_rate:
        subset = [ x for x in files if "_lr"+str(lr) in x]
        keep = keep + subset
    files = keep

if len(files) == 0:
    print( "No files found" )
    sys.exit(1)
    
def extract(fn):
    lb = ""
    hu = ""
    op = ""
    lr = ""
    do = ""
    try:
        bits = re.search('l(.*)_h', fn)
        lb = bits.group(1) 
    except:
        pass
        
    try:
        bits = re.search('h(.*?)_', fn)
        hu = bits.group(1) 
    except:
        pass

    try:
        bits = re.search('o(.*?)_', fn)
        op = bits.group(1) 
    except:
        pass

    try:
        bits = re.search('lr(.*?)_', fn)
        lr = bits.group(1) 
    except:
        pass
        
    try:
        bits = re.search('do(.*?)_', fn)
        do = bits.group(1) 
    except:
        pass

    return [lb, hu, op, lr, do]

acc = pd.DataFrame( columns=["Step"] )
for fn in files:
    if not "eval-tag-accuracy" in fn:
        continue

    lb, hu, op, lr, do = extract( fn )
    
    lbl = hu+" "+op+" "+lr+" "+do
    print( lbl )
    nxt = pd.read_csv( fn )
    nxt = nxt[ (nxt["Step"] <= 300000) ]
    nxt = nxt.drop(columns=["Wall time"])
    nxt = nxt.rename( index=str, columns={"Step": "Step", "Value": lbl} )
    acc = pd.merge( left=acc, right=nxt, left_on='Step', right_on='Step', how='outer' )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
acc.plot( x="Step", ax=ax )
ax.set_title( lb + " / eval-tag-accuracy" )
ax.set_ylim( (0.5,1) )
ax.set_xlim( (0,300000+10000) )
ax.legend(labelspacing=0.2, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig("curves_128x64_eval_acc.png", dpi=144)

# -----

acc = pd.DataFrame( columns=["Step"] )
for fn in files:
    if not "eval-tag-average_loss" in fn:
        continue

    lb, hu, op, lr, do = extract( fn )
        
    lbl = hu+" "+op+" "+lr+" "+do
    print( lbl )
    nxt = pd.read_csv( fn )
    nxt = nxt[ (nxt["Step"] <= 300000) ]
    nxt = nxt.drop(columns=["Wall time"])
    nxt = nxt.rename( index=str, columns={"Step": "Step", "Value": lbl} )
    acc = pd.merge( left=acc, right=nxt, left_on='Step', right_on='Step', how='outer' )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
acc.plot( x="Step", ax=ax )
ax.set_title( lb + " / eval-tag-average_loss" )
ax.set_ylim( (0,4) )
ax.set_xlim( (0,300000+10000) )
ax.legend(labelspacing=0.2, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig("curves_128x64_eval_loss.png", dpi=144)

# ----

plt.show(block=True)

