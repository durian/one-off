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
lENGINE_TYPE_h256xh125_Adam_lr0.0001_run_eval-tag-accuracy.csv
lENGINE_TYPE_h400xh400_run_eval-tag-accuracy.csv
lENGINE_TYPE_h800xh800_run_eval-tag-accuracy.csv
petber@ITE11527:~/Documents/HH/Health/tensorboard o
'''

# Hard coded, which is BORING, but it is a one-off thing (I think)
acc = pd.read_csv("lENGINE_TYPE_h100x100a_run_eval-tag-accuracy.csv")
acc = acc.drop(columns=["Wall time"])
acc = acc.rename( index=str, columns={"Step": "Step", "Value": "100x100"} )

nxt = pd.read_csv("lENGINE_TYPE_h200x200_run_eval-tag-accuracy.csv")
nxt = nxt.drop(columns=["Wall time"])
nxt = nxt.rename( index=str, columns={"Step": "Step", "Value": "200x200"} )

print( acc.head(4) )
print( nxt.head(4) )

df = pd.merge( left=acc, right=nxt, left_on='Step', right_on='Step', how='outer' )

print( df.head )

for fn, lbl in [ ["lENGINE_TYPE_h256xh125_Adam_lr0.0001_run_eval-tag-accuracy.csv", "256x125"],
                 ["lENGINE_TYPE_h400xh400_run_eval-tag-accuracy.csv", "400x400"],
                 ["lENGINE_TYPE_h800xh800_run_eval-tag-accuracy.csv", "800x800"] ]:
    print( fn, lbl )
    nxt = pd.read_csv( fn )
    nxt = nxt.drop(columns=["Wall time"])
    nxt = nxt.rename( index=str, columns={"Step": "Step", "Value": lbl} )
    df = pd.merge( left=df, right=nxt, left_on='Step', right_on='Step', how='outer' )

print( df.head )

df.plot( x="Step" )
plt.show(block=True)
