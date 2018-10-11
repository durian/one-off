from glob import glob
import os
import sys
import random
import argparse
import shutil

'''
Program to take the images and annotations from the labelImg
progran, and prepare a directory with data suitable for YoloV3.

Creates train, test and hold-out data sets, and a names and data file
for Yolo. The network definition (filters, classes) still needs to be
adjusted by hand.

All files will be copied and created inside the basename directory.
'''

parser = argparse.ArgumentParser()
parser.add_argument( '-b', "--basename", type=str, default="exp0",
                     help='Basename' )
#parser.add_argument( '-d', "--dirname", type=str, default="exp0/",
#                     help='Output directory name' )
parser.add_argument( '-f', "--force", action='store_true', default=False,
                     help='Force if directory exists' )
parser.add_argument( '-y', "--yolocfg", type=str, default="cfg/yolov3.cfg",
                     help='Output directory name' )
args = parser.parse_args()

# Use dirname as a basename for the names and data file?
if args.basename[:-1] == "/":
    args.basename = args.basename[:-1]

dirname        = args.basename + "/"
names_filename = args.basename + ".names"
data_filename  = args.basename + ".data"

filenames          = glob( "2*.txt" )
classes_filename   = "classes.txt"     #from the labelimg program
yolov3cfg_filename = args.yolocfg #"cfg/yolov3.cfg" 

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
'''
# maybe not in this script
import cv2
image = cv2.imread(filename) 
small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
cv2.imshow("small image",small)
cv2.imwrite('s.jpg',small)
'''
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
  classes = 1  
  train   = train.txt  
  valid   = test.txt  
  names   = obj.names  
  backup  = backup/
'''
with open(dirname+data_filename, "w") as f: 
    f.write( "classes={}\n".format(classes_count) )
    f.write( "train={}\n".format(dirname+"train.txt") )
    f.write( "valid={}\n".format(dirname+"test.txt") )
    f.write( "names={}\n".format(dirname+names_filename) )
    f.write( "backup={}\n".format(dirname+"backup/") )
#
print( "set classes={}".format( classes_count ))
#print( "set filters={}".format( (classes_count+5)*5 )) #yolo v2
print( "set filters={}".format( (classes_count+5)*(9/3) )) #yolo v3
'''
look for the htree [yolo] sections:
  inside yolo section, change classes
  inside [convolutional] before yolo, change filters
'''
if yolov3cfg_filename:
    with open(yolov3cfg_filename) as f:
        lines = f.readlines()
        idx   = 0
        yolos = []
        for line in lines:
            if "[yolo]" in line:
                yolos.append(idx)
                print( "yolo: {}". format(idx) )
            idx += 1
        for yolo in yolos:
            idx  = yolo + 1
            done = False
            while not done:
                if "classes" in lines[idx]:
                    print( lines[idx] )
                    lines[idx] = "classes={}\n".format( classes_count )
                    print( lines[idx] )
                    done = True
                idx += 1
        # backwards for filters
        for yolo in yolos:
            idx  = yolo - 1
            done = False
            while not done:
                if "filters" in lines[idx]:
                    print( lines[idx] )
                    lines[idx] = "filters={}\n".format( int((classes_count+5)*(9/3)) )
                    print( lines[idx] )
                    done = True
                idx -= 1
        #
        with open( dirname+args.basename+".cfg", "w" ) as f:
            for line in lines:
                f.write( line )
with open( dirname+args.basename+"_test.sh", "w" ) as f:
    for filename in hout_data:
        f.write( "./darknet detector test {} {} {} {}\n".format( dirname+data_filename,
                                                                 dirname+args.basename+".cfg", #maybe not specified
                                                                 dirname+"backup/",
                                                                 filename
        ))
        

# loop over hout.txt and prepare a batch script?
# ./darknet detector test exp0/config.txt exp0.cfg exp0/backup/exp0.backup exp0/20180905143500+0200-snapshot.jpg -thresh .1
#
#./darknet detector test kip0/kip0.data kip0/kip0.cfg kip0/backup/kip0.backup t_20181010173000+0200-snapshot.jpg -thresh 0.01
#cp predictions.jpg predictions3.jpg
