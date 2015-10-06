# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:43:10 2014

@author: htelg
"""

import numpy as np
import pandas as pd
import datetime
from atmPy import sizedistribution



def read_csv(fname, norm2time = True, norm2flow = True):
    uhsas = _readFromFakeXLS(fname)
#     return uhsas
    sd,hk = _separate_sizedist_and_housekeep(uhsas, norm2time = norm2time, norm2flow = norm2flow)
#     return sd,hk
    bins = _get_bins(sd)
#     return bins
    dist = sizedistribution.SizeDist_TS(sd,bins,"numberConcentration")
    return dist, hk

def _readFromFakeXLS(fname):
    """reads and shapes a XLS file produced by the uhsas instrument"""
    fr = pd.read_csv(fname, sep='\t')
    newcolname = [fr.columns[e] + ' ' + str(fr.values[0][e]) for e, i in enumerate(fr.columns)]
    fr.columns = newcolname
    fr = fr.drop(fr.index[0])
    bla = pd.Series(fr['Date -'].values + ' ' + fr['Time -'].values)
#     return bla
    fr.index = bla.map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S.%f'))
    fr = fr.drop(['Date -', 'Time -'], axis=1)
    return fr

def _separate_sizedist_and_housekeep(uhsas, norm2time = True, norm2flow = True):
    """Beside separating size distribution and housekeeping this
    function also converts the data to a numberconcentration (#/cc)

    Parameters
    ----------
    uhsas: pandas.DataFrame"""

    sd = uhsas.copy()
    hk = uhsas.copy()
#     return sd,hk
    k = sd.keys()
    where = np.argwhere(k == 'Valve 0=bypass') + 1
    khk = k[: where]
    sd = sd.drop(khk, axis=1)
    hsd = k[where:]
    hk = hk.drop(hsd, axis=1)
#     return sd,hk
    hk['Sample sccm'] = hk['Sample sccm'].astype(float)

    hk['Accum. Secs'] = hk['Accum. Secs'].astype(float)

    # normalize to time and flow
    if norm2time:
        sd = sd.mul(1 / hk['Accum. Secs'], axis = 0 )
    if norm2flow:
        sd = sd.mul(60./hk['Sample sccm'], axis = 0 )
    return sd,hk

def _get_bins(frame, log=False):
    """
    get the bins from the column labels of the size distribution DataFrame.
    """
    frame = frame.copy()

    bins = np.zeros(frame.keys().shape[0]+1)
    for e, i in enumerate(frame.keys()):
        bin_s, bin_e = i.split(' ')
        bin_s = float(bin_s)
        bin_e = float(bin_e)
        bins[e] = bin_s
        bins[e+1] = bin_e
    return bins #binCenters