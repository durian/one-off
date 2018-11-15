#! python3
#
import datetime
import random
import sys
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Classes & Funxions
# -----------------------------------------------------------------------------

'''
How to determine breakdown?
'''
class Part(object):
    def __init__(self, n, pd):
        self.name = n
        self.prod_date = pd
        self.fgrp4 = 9999
        self.warranty_months = 0
        self.fail_pm = 1 #breakdown per 1000
        self.fail_date = None #based on truck prod_date
    def __str__(self):
        return self.name+","+str(self.prod_date)
    def set_warranty(self, m):
        self.warranty_months = m # number of months
    def set_fail_pm(self, b):
        self.fail_pm = b
    def set_fgrp4(self, f):
        self.fgrp4 = f
    def calc_breakdown(self):
        #random.randint(); endpoints included
        r = random.randint(1, 1000)
        if self.fail_pm >= r:
            #break, set the date
            self.fail_date = self.prod_date + datetime.timedelta(days=random.randint(1,365))
            #print( "->", self.name, self.fail_pm, r, str(self.fail_date) )
        
class Turbo(Part):
    def __init__(self, n, pd):
        # Data sets
        Part.__init__(self, n, pd)
        Part.set_fgrp4(self, 2846)
        
class Brake(Part):
    def __init__(self, n, pd):
        # Data sets
        Part.__init__(self, n, pd)
        Part.set_fgrp4(self, 1200)

class Battery(Part):
    def __init__(self, n, pd):
        # Data sets
        Part.__init__(self, n, pd)
        Part.set_fgrp4(self, 3111)
        Part.set_fail_pm(self, 20)

class Compressor(Part):
    def __init__(self, n, pd):
        # Data sets
        Part.__init__(self, n, pd)
        Part.set_fail_pm(self, 10)
        Part.set_fgrp4(self, 4004)

# Assembly lines
def truck_normal(pd=datetime.date(2010, 1, 1)):
    return [ Turbo("Tu", pd), Brake("Br", pd), Compressor("Co", pd), Battery("Ba", pd) ]

class Truck(object):
    def __init__(self, i, pd=datetime.date(2010, 1, 1)):
        self.id = "1{:04d}".format(i)
        self.prod_date = pd
        self.parts = self.new_parts()
        self.fail_date = None
    def new_parts(self):
        return truck_normal(self.prod_date) # get parts from assembly line
    def inc_pd(self, d): # add d days to prod date
        print( "DEPRECATED" )
        self.prod_date = self.prod_date + datetime.timedelta(days=d)
        self.parts = self.new_parts() #redo because of dates
    def __str__(self):
        return "T-"+str(self.id)+","+str(self.prod_date)+","+str(self.get_breakdown_dates())
    def calc_breakdowns(self):
        for part in self.parts:
            part.calc_breakdown()
        # remove the parts that don't break down, and sort on date
        self.parts = [ p for p in self.parts if p.fail_date != None]
        self.parts.sort(key=lambda x: x.fail_date, reverse=False)
    def get_breakdown_dates(self): #returns dates
        return [ str(p.fail_date) for p in self.parts if p.fail_date != None]
    def get_breakdowns(self):
        return [ p for p in self.parts if p.fail_date != None]
    def get_claims(self):
        bds = self.get_breakdowns() #List of broken down parts
        ans = []
        for bd in bds:
            #print( self.id, self.prod_date, bd.fgrp4, bd.fail_date )
            ans.append( [self.id, str(self.prod_date), bd.fgrp4, str(bd.fail_date)] )
        return ans
    
# -----------------------------------------------------------------------------

def dates(sd=datetime.date(2010, 1, 1), num=365):
    '''
    Return the working dates between start date and start date
    plus num days.
    '''
    work_dates = []
    for d in range(0, num):
        dt = sd + datetime.timedelta(days=d)
        if dt.weekday() > 4: # 5, 6 is sat, sun
            continue
        isoc = dt.isocalendar() #tuple containing week nr
        if isoc[1] in [28, 29, 30, 51, 52, 53]:
            continue #holidays
        work_dates.append( dt )
    return work_dates
        

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    #print( "Use this for import" )
    c = 0
    broken = []
    good = []
    prod_dates = dates(sd=datetime.date(2010, 1, 1), num=365*2)
    for i in range(0, 100000): #create 1000 trucks
        t = Truck(c, prod_dates[random.randint(0, len(prod_dates)-1)])
        #t.inc_pd( random.randint(0, 365) ) #in a one year period (july less?)
        t.calc_breakdowns()
        if t.get_breakdowns():
            #print( t )
            #print()
            broken.append( t )
        else:
            good.append( [t.id, t.prod_date] )
        c += 1
    #print( broken )
    print( len(good), len(broken) )
    #
    good_df = pd.DataFrame( good, columns=["VEHICLE_ID", "VEH_ASSEMB_DATE"] )
    good_sums = good_df.groupby('VEH_ASSEMB_DATE').size()
    good_cumsums = good_df.groupby('VEH_ASSEMB_DATE').size().cumsum() # VEH_AGE_CLAIM
    print( good_cumsums.tail() )
    #sys.exit(1)
    # generate plots?
    broken.sort(key=lambda x: x.prod_date, reverse=False)
    all_claims = []
    for t in broken:
        #print( t )
        claims = t.get_claims()
        for c in claims:
            all_claims.append( c )
    #print( all_claims )
    df = pd.DataFrame( all_claims, columns=["VEHICLE_ID","VEH_ASSEMB_DATE","FGRP4","CLAIM_DATE"] )
    print( df )
    #print( "VEHICLE_ID,VEH_PROD_DATE,FGRP4,CLAIM_DATE" )
    #for c in all_claims:
    #    print( "{:s},{:s},{:d},{:s}".format(*c) )
    sums = df.groupby('VEH_ASSEMB_DATE').size()
    print( sums )
    cumsums = df.groupby('VEH_ASSEMB_DATE').size().cumsum() # VEH_AGE_CLAIM
    print( cumsums )
    #
    #        subset_p = subset_p.resample('w').mean() #.fillna(0)
    #df1 = df.set_index(["VEH_ASSEMB_DATE"])
    #print( df1 )
    sums.index = pd.to_datetime(sums.index)
    print(sums)
    sums = sums.resample('M').sum() #or cumsum()
    print( sums )
    #
    #sums.plot()
    fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
    c = '#4A7BC9'
    ax0.vlines(x=sums.index,
               ymin=0,
               ymax=sums.values, color=c, alpha=0.5,
               label="Normalised warranty vol")
    plt.show(block=True)
