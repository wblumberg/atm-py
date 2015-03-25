# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:10:55 2014

@author: htelg
"""
import pandas as pd
import datetime
import os
import pylab as plt

def read_housekeeping(fname):
    """Reads housekeeping file (fname; csv-format) returns a pandas data frame instance."""
    df = pd.read_csv(fname,error_bad_lines=False)

#    data = df.values
#    dateString = fname.split('_')[0]
    dt = datetime.datetime.strptime('19700101', "%Y%m%d") - datetime.datetime.strptime('19040101', "%Y%m%d") 
    dts = dt.total_seconds()
    dtsPlus = datetime.timedelta(hours = 8).total_seconds()
    # Time_s = data[:,0]
    # data = data[:,1:]
    df.index = pd.Series(pd.to_datetime(df.Time_s-dts-dtsPlus, unit = 's'), name = 'Time_UTC')
    return df

def read_housekeeping_allInFolder(concatWithOther = False, other = False, skip=[]):
    """Read all housekeeping files in current folder and concatinates them.
    Output: pandas dataFrame instance
    Parameters
        concatWithOther: bool, if you want to concat the created data with an older set given by other
        other: dataframe to concat the generated one
        skip: list of file which you want to exclude"""
        
    files = os.listdir('./')
    if concatWithOther:
        counter = True
        hkdf = other.copy()
    else:
        counter = False
    for e,i in enumerate(files):
        if 'HK.csv' in i:
            if i in skip:
                continue
            hkdf_tmp = read_housekeeping(i)
            if not counter:
                hkdf = hkdf_tmp
            else:
                hkdf = pd.concat([hkdf,hkdf_tmp])
            counter = True
    return hkdf
    
def plot_all(hk):
    for k in hk.keys():
        f,a = plt.subplots()
        hk[k].plot(ax = a)
        a.set_title(k)