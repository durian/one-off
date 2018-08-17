import pandas as pd
import numpy as np
from collections import Counter
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

files = [
    [ "lENGINE_TYPE_h400xh400_run_eval-tag-average_loss.csv", "400x400 Default" ],
    [ "lENGINE_TYPE_h800xh800_run_eval-tag-average_loss.csv", "800x800 Default" ],
    [ "lENGINE_TYPE_h128x64_oAdam_lr0.0001_b0_run_eval-tag-average_loss.csv", "128x64 Adam 0.0001"],
    [ "lENGINE_TYPE_h128x64_oAdam_lr0.001_b0_run_eval-tag-average_loss.csv", "128x64 Adam 0.001"], 
    [ "lENGINE_TYPE_h256x128_oAdam_lr0.0001_b0_run_eval-tag-average_loss.csv", "256x128 Adam 0.0001"], 
    [ "lENGINE_TYPE_h256x128_oAdam_lr0.001_b0_run_eval-tag-average_loss.csv", "256x128 Adam 0.001"], 
    [ "lENGINE_TYPE_h512x256_oAdam_lr0.0001_b0_run_eval-tag-average_loss.csv", "512x256 Adam 0.0001"]
]

# Crude, but ok...
if len(files) < 2:
    sys.exit(1)
    
# Hard coded, which is BORING, but it is a one-off thing (I think)
fn, lbl = files[0]
acc = pd.read_csv( fn )
acc = acc[ (acc["Step"] <= 300000) ]
acc = acc.drop(columns=["Wall time"])
acc = acc.rename( index=str, columns={"Step": "Step", "Value": lbl} )

# Merge the rest into this frame
for fn, lbl in files[1:]:
    print( fn, lbl )
    nxt = pd.read_csv( fn )
    nxt = nxt[ (nxt["Step"] <= 300000) ]
    nxt = nxt.drop(columns=["Wall time"])
    nxt = nxt.rename( index=str, columns={"Step": "Step", "Value": lbl} )
    acc = pd.merge( left=acc, right=nxt, left_on='Step', right_on='Step', how='outer' )

print( acc.head )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
acc.plot( x="Step", ax=ax )
#ax.set_ylim( (0.5,1) )
ax.set_xlim( (0,300000+10000) )
fig.savefig("curves_eval_loss.png", dpi=144)
plt.show(block=True)

