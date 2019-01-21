#!/usr/bin/env python3
#
import re
import getopt, sys, os
import subprocess
import shlex
import signal
import argparse
import os
import shutil
import glob
import datetime
    
parser = argparse.ArgumentParser()
parser.add_argument( '-c', "--camera", type=str, default="Camera-84", help='Camera-ID folder' )
args = parser.parse_args()


'''
args = shlex.split(self.param['cmd'])
self.p = subprocess.Popen(args,\
                          shell=False,\
                          bufsize=-1,\
                          stdout=self.param['stdout_fd'],\
                          stderr=self.param['stderr_fd'],\
                          cwd=self.param['cwd'])
'''

#filenames = glob.glob( "timelapse_files/*.ts" )
folder = args.camera #"Camera-84" # subdir with ts files, move bb files to it?
filenames = glob.glob( "timelapse_new/"+folder+"/*.ts" )
for filename in filenames:
    if os.path.exists( filename+"_PROCESSED" ):
        #print( "Already processed" )
        #continue  in timelapse_files?
        pass
    
    print( "Moving", filename, "to timelapse_files/" )
    filename_base = os.path.basename( filename )
    shutil.copy2( filename, "timelapse_files/" )
    # /Users/petber/Documents/HH/Videquus/videquus-IR
    cwd = os.getcwd()
    print( cwd )
    cmd = "sh ./docker_run_timelapse.sh -v "+cwd+"/results:/usr/src/app/results -v "+cwd+"/timelapse_files:/usr/src/app/timelapse_files"
    cmd_args = shlex.split( cmd )
    print( cmd_args )

    utc_datetime = datetime.datetime.utcnow()
    #utc_datetime += datetime.timedelta(seconds=3) #add the 3 secs lag (natte vingerwerk...)
    utc_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print( utc_datetime ) #seems 3 secs too early :-) but minute is prolly enough, but could be 1 off...
    # Run
    with open("batch_log.txt", "a") as stdout_fd:
        p = subprocess.Popen(cmd_args,
                             shell=False,
                             bufsize=-1,
                             stdout=stdout_fd, 
                             #stderr=self.param['stderr_fd'],
                             #cwd="."
        )
        p.wait()
    # (re)move ts files. hwo to determine output map?
    print( "Removing", "timelapse_files/" + filename_base )
    os.remove( "timelapse_files/" + filename_base )
    print( "Moving", filename, "to", filename+"_PROCESSED" ) 
    shutil.move( filename, filename+"_PROCESSED" ) # hmm, move to folder?
    print( "Ready" )
