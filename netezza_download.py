# -*- coding: utf-8 -*-
"""
(c) PJB 2018
"""
import pyodbc
import csv
 
max_lines  = 100000 # per file
lines      =      0
curr_csv   =      0
max_files  =    999 # max number of files to  create
csv_file   = None
csv_writer = None
columns    = []
#
# This is "version 3" (note filename later on)
#table      = "VTC_F_TEA2_PLUS_MULTI_P1FWM_FLATTEND_TRAIN_TEST_STEERING_PUMP"
table      = "VTC_F_TEA2_PLUS_MULTI_P1FW6_FLATTENED" # only 3038 histograms?
# This is versions 1 and 2:
#table = "VTC_F_TEA2_PLUS_MULTI_P1FWM_FLATEN"
 
# Start and end dates
dt_0 = "20100101"
dt_1 = "20190101"
prefix = "P1FW6_20181008"
 
# pyodbc connection string
conn = pyodbc.connect("DRIVER={NetezzaSQL};SERVER=edwqa.volvo.net;DATABASE=PROD_PMA_DL;UserName=xxxx;Password=xxxx;Timeout=120")
 
print( conn )
 
def create_csv():
    global csv_file, curr_csv, lines, csv_writer, columns
    if csv_file:
        csv_file.close()
        curr_csv += 1
        lines = 0
        if curr_csv >= max_files:
            print( "max_files reached" )
            return False
    csv_fn = "C:\\Users\\a282102\\Downloads\\"+prefix+"_"+dt_0+"-"+dt_1+"_"+str(curr_csv)+".csv"
    print( csv_fn )
    csv_file = open( csv_fn, "w" )
    csv_writer = csv.writer( csv_file, delimiter=";", lineterminator='\n' )
    csv_writer.writerow( columns )
    return True
 
def csv_write( row ):
    global csv_file, curr_csv, lines, csv_writer
    csv_writer.writerow( row )
    lines += 1
    if lines >= max_lines:
        return create_csv()
    return True
 
# Define Cursor
cursor = conn.cursor()
#cursor.execute("select count(*) from "+table+" where SEND_DATE >= to_date("+dt_0+", 'YYYYMMDD') and SEND_DATETIME < to_date("+dt_1+", 'YYYYMMDD');")
cursor.execute("select count(*) from "+table+";")
 
while True:
    row = cursor.fetchone()
    if not row:
        break
    print( row )
    #csv_write( row )
   
# Execute SQL statement and store result in cursor
cursor.execute("select * from "+table+" order by VEHICLE_ID desc;")
columns = [column[0] for column in cursor.description]
print( columns )
create_csv()
 
#csv_writer.writerow( columns )
 
# Display the content of cursor
while True:
    row=cursor.fetchone()
    if not row:
        break
    #print(row)
    if not csv_write(row):
        break
   
cursor.close()
conn.close()
csv_file.close()
print( "Ready" )
 
# 1 million lines = 90MB data ... 800 * 90 MB = 72 GB
# limit 2016-01 to now, or 2 years back
