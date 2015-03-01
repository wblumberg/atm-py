# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 17:04:22 2015

@author: htelg
"""

import datetime
import numpy as np

def time_mac2dt(secs, timezone = 0, verbose = False, dateZero = "19040101" ):
    """ creates a dateTime opject from a timestamp which represents seconds from 1904-01-01
    parameters:
    \t secs:\t array-like opject of seconds (float) since 1904-01-01"""
    
    d0 = datetime.datetime.strptime(dateZero, "%Y%m%d")
    out = []
    for t in secs:
        out.append(d0 + datetime.timedelta(seconds = t))
    if verbose:
        print out[0].strftime("%Y-%m-%d_%H:%M:%S:%f")
    return np.array(out)