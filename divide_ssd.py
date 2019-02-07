from glob import glob
import os
import sys
import random
import argparse
import shutil

'''
Program to take the images and annotations from the labelImg
program, and prepare a directory with data suitable for ssd_keras.

Creates train, test and hold-out data sets, and a names and data file.

All files will be copied and created inside the basename directory.
'''

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

# Image files are in current directory
filenames          = glob( "2*.txt" )  #annotations, only use files which have a .txt file
classes_filename   = "classes.txt"     #from the labelimg program

# The classes.txt/obj.names file needs to be copied by hand!
jpg_filenames = []
for filename in filenames:
    jpg_filename = filename[:-3]+"jpg"
    print( filename, jpg_filename )
    jpg_filenames.append( jpg_filename )

count = len(jpg_filenames)
count_train = int(count * 0.7)
count_test  = int(count * 0.2 )
count_hout  = count - count_train - count_test  # hout = hold out
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
shutil.copy2( classes_filename, dirname+names_filename ) 
'''
# maybe not in this script
import cv2
image = cv2.imread(filename) 
small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
cv2.imshow("small image",small)
cv2.imwrite('s.jpg',small)
'''
# What is format needed/possible for ssd_keras?
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
with open( dirname+args.basename+"_test.sh", "w" ) as f:
    f.write( "WEIGHTS={}/backup/{}.backup\n".format(dirname, args.basename) ) #backup/kip0.backup
    for filename in hout_data:
        f.write( "./darknet detector test {} {} {} {}\n".format( dirname+data_filename,
                                                                 dirname+args.basename+".cfg", #maybe not specified
                                                                 "${WEIGHTS}",
                                                                 filename
        ))
'''        

