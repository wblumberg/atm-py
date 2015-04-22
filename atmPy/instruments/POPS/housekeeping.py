# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:10:55 2014

@author: htelg
"""
import pandas as pd
import datetime
import os
import pylab as plt
from atmPy import housekeeping
from atmPy.tools import conversion_tools as ct


def _read_housekeeping(fname):
    """Reads housekeeping file (fname; csv-format) returns a pandas data frame instance."""
    try:
        df = pd.read_csv(fname, error_bad_lines=False)
    except ValueError:
        return False
#    data = df.values
#    dateString = fname.split('_')[0]
    dt = datetime.datetime.strptime('19700101', "%Y%m%d") - datetime.datetime.strptime('19040101', "%Y%m%d") 
    dts = dt.total_seconds()
    # todo: (low) what is that delta t for, looks fishi (Hagen)
    dtsPlus = datetime.timedelta(hours=0).total_seconds()
    # Time_s = data[:,0]
    # data = data[:,1:]
    df.index = pd.Series(pd.to_datetime(df.Time_s-dts-dtsPlus, unit = 's'), name = 'Time_UTC')
    if 'P_Baro' in df.keys():
        df['barometric_pressure'] = df.P_Baro
        df.drop('P_Baro', 1, inplace=True)
        df['altitude'] = ct.p2h(df.barometric_pressure)
    return housekeeping.HouseKeeping(df)


def read_housekeeping(fname):
    """
    Parameters
    ----------
    fname: string or list of strings.

    Returns
    -------
    HouseKeeping instance
    """
    fname = os.listdir()
    # fname = '20150419_000_POPS_HK.csv'
    first = True
    if type(fname).__name__ == 'list':
        for file in fname:
            if 'HK.csv' in file:
                hktmp = _read_housekeeping(file)
                if not hktmp:
                    print('%s is empty ... next one' % file)
                elif first:
                    data = hktmp.data.copy()
                    first = False
                    continue

                else:
                    data = pd.concat((data, hktmp.data))
                    hk = housekeeping.HouseKeeping(data)
    else:
        hk = _read_housekeeping(fname)
    return hk

# todo: (low) this has never been actually implemented
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

