import pandas as pd
import numpy as np
from collections import Counter
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

#trainset = pd.read_csv("trainset_dnn.csv")
#print( trainset.shape )
#testset = pd.read_csv("testset_dnn.csv")
#valset = pd.read_csv("validationset_dnn.csv")

'''
lENGINE_TYPE_h100x100a_run_eval-tag-accuracy.csv
lENGINE_TYPE_h200x200_run_eval-tag-accuracy.csv
lENGINE_TYPE_h256xh128_Adam_lr0.0001_run_eval-tag-accuracy.csv
lENGINE_TYPE_h400xh400_run_eval-tag-accuracy.csv
lENGINE_TYPE_h800xh800_run_eval-tag-accuracy.csv
petber@ITE11527:~/Documents/HH/Health/tensorboard o
'''

files = [ #["lENGINE_TYPE_h256xh128_Adam_lr0.0001_run_eval-tag-accuracy.csv", "256x128 Adam 0.0001"],
    #["lENGINE_TYPE_h400xh400_run_eval-tag-accuracy.csv", "400x400"],
    #["lENGINE_TYPE_h800xh800_run_eval-tag-accuracy.csv", "800x800"],
    ["lENGINE_TYPE_h128x64_oAdam_lr0.0001_b0_run_eval-tag-accuracy.csv",  "128x64  Adam 0.0001"],
    ["lENGINE_TYPE_h128x64_oAdam_lr0.001_b0_run_eval-tag-accuracy.csv",   "128x64  Adam 0.001"],
    ["lENGINE_TYPE_h256x128_oAdam_lr0.0001_b0_run_eval-tag-accuracy.csv", "256x128 Adam 0.0001"],
    ["lENGINE_TYPE_h256x128_oAdam_lr0.001_b0_rrun_eval-tag-accuracy.csv", "256x128 Adam 0.001"],
    ["lENGINE_TYPE_h512x256_oAdam_lr0.0001_b0_run_eval-tag-accuracy.csv", "512x256 Adam 0.0001"]
]

# Crude, but ok...
if len(files) < 2:
    sys.exit(1)
    
# Hard coded, which is BORING, but it is a one-off thing (I think)
fn, lbl = files[0]
acc = pd.read_csv( fn )
acc = acc.drop(columns=["Wall time"])
acc = acc.rename( index=str, columns={"Step": "Step", "Value": lbl} )

fn, lbl = files[1]
nxt = pd.read_csv( fn )
nxt = nxt.drop(columns=["Wall time"])
nxt = nxt.rename( index=str, columns={"Step": "Step", "Value": lbl} )

print( acc.head(4) )
print( nxt.head(4) )

df = pd.merge( left=acc, right=nxt, left_on='Step', right_on='Step', how='outer' )

print( df.head )

for fn, lbl in files[2:]:
    print( fn, lbl )
    nxt = pd.read_csv( fn )
    nxt = nxt.drop(columns=["Wall time"])
    nxt = nxt.rename( index=str, columns={"Step": "Step", "Value": lbl} )
    df = pd.merge( left=df, right=nxt, left_on='Step', right_on='Step', how='outer' )

print( df.head )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
df.plot( x="Step", ax=ax )
ax.set_ylim( (0.5,1) )
fig.savefig("curves.png", dpi=144)
plt.show(block=True)
