from glob import glob
import os
import sys
import random
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument( '-b', "--basename", type=str, default="exp0",
                     help='Basename' )
#parser.add_argument( '-d', "--dirname", type=str, default="exp0/",
#                     help='Output directory name' )
parser.add_argument( '-f', "--force", action='store_true', default=False,
                     help='Force if directory exists' )
args = parser.parse_args()

# Use dirname as a basename for the names and data file?
if args.basename[:-1] == "/":
    args.basename = args.basename[:-1]

dirname        = args.basename + "/"
names_filename = args.basename + ".names"
data_filename  = args.basename + ".data"

filenames          = glob( "2*.txt" )
classes_filename   = "classes.txt"     #from the labelimg program
yolov3cfg_filename = "cfg/yolov3.cfg"  #not used

# The classes.txt/obj.names file needs to be copied by hand!

jpg_filenames = []
for filename in filenames:
    jpg_filename = filename[:-3]+"jpg"
    print( filename, jpg_filename )
    jpg_filenames.append( jpg_filename )

count = len(jpg_filenames)
count_train = int(count * 0.7)
count_test  = int(count * 0.2 )
count_hout  = count - count_train - count_test
print( count, count_train, count_test, count_hout )
random.shuffle( jpg_filenames )
train_data = jpg_filenames[:count_train]
test_data  = jpg_filenames[count_train:count_train+count_test]
hout_data  = jpg_filenames[count_train+count_test:]
#
#print( train_data, test_data, hout_data )
#sys.exit(1)
#
classes_count = 0
with open( classes_filename, "r" ) as f:
    lines = f.readlines()
    classes_count = len( lines )
print( "classes", classes_count )
#
try:
    os.makedirs( dirname, exist_ok=False )
except OSError:
    if not args.force:
        print( "Already exists." )
        sys.exit(1)
#
try:
    os.makedirs( dirname+"backup/", exist_ok=False )
except OSError:
    print( "NB: backup/ already exists." )
#
shutil.copy2( classes_filename, dirname+names_filename ) 
with open(dirname+"train.txt", "w") as f:
    for filename in train_data:
        shutil.copy2( filename, dirname+filename )
        txt_filename = filename[:-3]+"txt" #double, improve this earlier
        shutil.copy2( txt_filename, dirname+txt_filename )
        f.write( "{}{}\n".format(dirname, filename) )
with open(dirname+"test.txt", "w") as f:
    for filename in test_data:
        shutil.copy2( filename, dirname+filename )
        txt_filename = filename[:-3]+"txt"
        shutil.copy2( txt_filename, dirname+txt_filename )
        f.write( "{}{}\n".format(dirname, filename) )
with open(dirname+"hout.txt", "w") as f:
    for filename in hout_data:
        shutil.copy2( filename, dirname+filename )
        txt_filename = filename[:-3]+"txt"
        shutil.copy2( txt_filename, dirname+txt_filename )
        f.write( "{}{}\n".format(dirname, filename) )
'''
For YoloV3:
classes= 1  
train  = train.txt  
valid  = test.txt  
names = obj.names  
backup = backup/
'''
with open(dirname+data_filename, "w") as f: 
    f.write( "classes={}\n".format(classes_count) )
    f.write( "train={}\n".format(dirname+"train.txt") )
    f.write( "valid={}\n".format(dirname+"test.txt") )
    f.write( "names={}\n".format(dirname+classes_filename) )
    f.write( "backup={}\n".format(dirname+"backup/") ) # needs to be created
'''
Line 3: set batch=64, this means we will be using 64 images for
every training step

Line 4: set subdivisions=8, the batch will be divided by 8 to decrease
GPU VRAM requirements. If you have a powerful GPU with loads of VRAM,
this number can be decreased, or batch could be increased. The
training step will throw a CUDA out of memory error so you can adjust
accordingly.

Line 244: set classes=1, the number of categories we want to detect

Line 237: set filters=(classes + 5)*5 in our case filters=30
'''
print( "set classes={}".format( classes_count ))
#print( "set filters={}".format( (classes_count+5)*5 )) #yolo v2
print( "set filters={}".format( (classes_count+5)*(9/3) )) #yolo v3

# loop over hout.txt and prepare a batch script?
# ./darknet detector test exp0/config.txt exp0.cfg exp0/backup/exp0.backup exp0/20180905143500+0200-snapshot.jpg -thresh .1
