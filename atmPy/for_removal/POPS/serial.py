import numpy as np
import pandas as pd

from atmPy.general import timeseries
from atmPy.aerosols.size_distr import sizedistribution
from atmPy.for_removal.POPS import calibration
from atmPy.tools import time_tools


def read_radiosonde_csv(fname, cal):
    """reads a csv file and returns a TimeSeries

    Parameters
    ----------
    fname: str
        Name of file to be opend
    calibration: str or calibration instance
        Either pass the name of the file containing the calibration data, or a calibration instance.

    """

    df = pd.read_csv(fname,header = 15)

    fkt = lambda x: x.lstrip(' ').replace(' ', '_')
    col_new = [fkt(i) for i in df.columns.values]
    df.columns = col_new

    time = df['date_[y-m-d_GMT]'] + df['time_[h:m:s_GMT]'] + '.' + df['milliseconds'].astype(str)
    df.index = pd.Series(pd.to_datetime(time, format = time_tools.get_time_formate()))

    df[df == 99999.000] = np.nan

    alt = df['GPS_altitude_[km]'].copy()
    df['Altitude'] = alt * 1e3
    df.rename(columns={'GPS_latitude':'Lat', 'GPS_longitude': 'Lon'}, inplace=True)

    bins = []
    for k in df.keys():
        if 'Bin' in k:
            bins.append(k)
    #         print(k)
#     print(bins)
    sd = df.loc[:,bins]

    hk = df.drop(bins, axis=1)

    hk = timeseries.TimeSeries(hk)
    hk.data.sort_index(inplace=True)
    hk.data.Altitude.interpolate(inplace=True)

#     fname_cal = '/Users/htelg/data/POPS_calibrations/150622_china_UAV.csv'
    cal = calibration.read_csv(cal)
    ib = cal.get_interface_bins(20)
    sd = sizedistribution.SizeDist_TS(sd, ib['binedges_v_int'].values.transpose()[0], 'numberConcentration')
    return sd,hk