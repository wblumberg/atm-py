# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:41:17 2015

@author: htelg
"""
import os
from struct import unpack, calcsize
import numpy as np

def load_raw(fname):
    ''' load a raw_file and returns a numpy array. 
    Note, these files have no x axes. The axes depends on the sampling rate of the particular POPS daughter board.
    Usually this is 4 MHz, however, better check!'''
    entry_format = '>h'
    entry_size = calcsize(entry_format)

    rein = open(fname, mode='rb')
    entry_count = int(os.fstat(rein.fileno()).st_size / entry_size)


    raus = np.zeros(entry_count)

    for e,r in enumerate(raus):
        record = rein.read(entry_size)
        entry, = unpack(entry_format, record)
    #     print entry


        if e == -1:
            # print entry
            break
        raus[e] = float(entry)

    rein.close()
    return raus