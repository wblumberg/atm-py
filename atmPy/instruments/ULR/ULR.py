# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:23:22 2015

@author: htelg
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

year = '2015'


def read_csv(fname, verbose=True):
    """Creates a single ULR instance from one file or a list of files.

    Arguments
    ---------
    fname: string or list
    """
    if type(fname).__name__ == 'list':
        first = True
        for file in fname:
            if file[0] != 'r':
                continue
            if verbose:
                print(file)
            ulrt = ULR(file, verbose=verbose)
            if first:
                ulr = ulrt
                first = False
            else:
                ulr.data = pd.concat((ulr.data, ulrt.data))
    else:
        ulr = ULR(fname, verbose=verbose)

    return ulr


# todo: see warning below
def _recover_Negatives(series, verbose=True):
    series = series.values
    where = np.where(series>2**16)
    series[where] = (series[where]*2**16)/2**16
    if where[0].shape[0] > 0:
        if verbose:
            warnings.warn("""This has to be checked!!! Dont know if i implemented it correctly! Arduino negatives become very large positive (unsigned longs)
           ;  recover using deliberate overflow""")


def _extrapolate(x, y):
    """This is a very simple extrapolation. 
    Takes: two series of the same pandas DataTable. x is most likely the index of the DataTable
    assumption: 
        - x is are the sampling points while y is the dataset which is incomplete and needs to be extrapolated
        - the relation ship is very close to linear
    proceedure:
        - takes the fist two points of y and performs a linear fit. This fit is then used to calculate y at the very first
          x value
        - similar at the end just with the last two points.
    returns: nothing. everthing happens inplace
    """
    
    xAtYnotNan = x.values[~np.isnan(y.values)][:2]
    YnotNan = y.values[~np.isnan(y.values)][:2]
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(xAtYnotNan,YnotNan)

    fkt = lambda x: intercept + (slope * x)
    y.values[0] = fkt(x.values[0])

    xAtYnotNan = x.values[~np.isnan(y.values)][-2:]
    YnotNan = y.values[~np.isnan(y.values)][-2:]
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(xAtYnotNan,YnotNan)

    fkt = lambda x: intercept + (slope * x)
    y.values[-1] = fkt(x.values[-1])
    
    return


def _simplefill(series):
    """Very simple function to fill missing values. Should only be used for 
    values which basically do not change like month and day. 
    Will most likely give strange results when the day does change.
    Returns: nothing everything happens inplace"""
    
    series.values[0] = series.dropna().values[0]
    series.values[-1] = series.dropna().values[-1]
    series.fillna(method='ffill', inplace = True)
    return


class ULR(object):
    def __init__(self, fname, verbose=True):
        self.verbose = verbose
        self.read_file(fname)
        # self.assureFloat()
        self.recover_negative_values()
        self.normalizeToIntegrationTime()
        self.set_dateTime()
        self.remove_data_withoutGPS()
        self.remove_unused_columns()


    def remove_data_withoutGPS(self, day='08', month='01'):
        """ Removes data from before the GPS is fully initiallized. At that time the Date should be the 8th of January.
        This is an arbitray value, which might change

        Arguments
        ---------
        day (optional): string of 2 digit integer
        month (optional): string of 2 digit integer
        """
        self.data = self.data[((self.data.Day != day) & (self.data.Month != month))]
    
    def read_file(self,fname):
        df = pd.read_csv(fname,
                         encoding="ISO-8859-1",
                         skiprows=16,
                         header=None,
                         error_bad_lines=False,
                         warn_bad_lines= False
                        )

        #### set column labels
        collabels = ['PhotoAsh', 
                     'PhotoBsh',
                     'PhotoCsh',
                     'PhotoDsh',
                     'PhotoA',
                     'PhotoB',
                     'PhotoC',
                     'PhotoD',
                     'Seconds',
                     'caseflag',
                     'var1',
                     'var2',
                     'var3']

        df.columns = collabels

        #### Drop all lines which lead to errors
        df = df.convert_objects(convert_numeric='force')
        df = df.dropna(subset=['Seconds'])
        # self.data = df
        # return
        df = df.astype(float)
        # self.data = df
        # return

        #### add extra columns
        df['time'] = np.nan
        df['azimuth'] = np.nan
        df['homep'] = np.nan
        df['MicroUsed'] = np.nan
        df['lat'] = np.nan
        df['lon'] = np.nan
        df['Te'] = np.nan
        df['GPSHr'] = np.nan
        df['MonthDay'] = np.nan
        df['Month'] = np.nan
        df['Day'] = np.nan
        df['GPSReadSeconds'] = np.nan
        df['HKSeconds'] = np.nan
        df['Yaw'] = np.nan
        df['Pitch'] = np.nan
        df['Roll'] = np.nan
        df['BaromPr'] = np.nan
        df['BaromTe'] = np.nan
        df['Modeflag'] = np.nan
        df['GateLgArr'] = np.nan
        df['GateShArr'] = np.nan
        df['PhotoOffArr'] = np.nan


        ##### Case 0
        case = np.where(df.caseflag.values == 0)
        df.azimuth.values[case] = df.var1.values[case]
        df.homep.values[case] = df.var2.values[case]
        df.MicroUsed.values[case] = df.var3.values[case]

        ##### Case 1
        case = np.where(df.caseflag.values == 1)
        df.lat.values[case] = df.var1.values[case]
        df.lon.values[case] = df.var2.values[case]
        df.Te.values[case] = df.var3.values[case]

        ##### Case 2
        case = np.where(df.caseflag.values == 2)
        df.GPSHr.values[case] = df.var1.values[case]
        df.MonthDay.values[case] = df.var2.values[case]
        df.GPSReadSeconds.values[case] = df.var3.values[case].astype(float) / 100.
        df.HKSeconds.values[case] = df.Seconds.values[case]

        ##### Case 3
        case = np.where(df.caseflag.values == 3)
        df.Yaw.values[case] = df.var1.values[case]
        df.Pitch.values[case] = df.var2.values[case]
        df.Roll.values[case] = df.var3.values[case]

        ##### Case 4
        case = np.where(df.caseflag.values == 4)
        df.BaromPr.values[case] = df.var1.values[case]
        df.BaromTe.values[case] = df.var2.values[case]
        df.Modeflag.values[case] = df.var3.values[case].astype(float) + 0.5

        ##### Case 5
        case = np.where(df.caseflag.values == 5)
        df.GateLgArr.values[case] = df.var1.values[case]
        df.GateShArr.values[case] = df.var2.values[case]
        df.PhotoOffArr.values[case] = df.var3.values[case]
        _simplefill(df.GateLgArr)
        _simplefill(df.GateShArr)
        self.data = df

    def assureFloat(self):
        """
        Note
        ----
        Ever now an then there is a line, which start with 'GPRMC'. This Line causes trouble. Therefore """

        where = np.where(self.data.PhotoAsh == 'GPRMC')
        self.data.PhotoAsh.values[where] = np.nan
        self.data.PhotoBsh.values[where] = np.nan
        self.data.PhotoCsh.values[where] = np.nan
        self.data.PhotoDsh.values[where] = np.nan

        self.data.PhotoA.values[where] = np.nan
        self.data.PhotoB.values[where] = np.nan
        self.data.PhotoC.values[where] = np.nan
        self.data.PhotoD.values[where] = np.nan

        self.data.PhotoAsh = self.data.PhotoAsh.values.astype(float)
        self.data.PhotoBsh = self.data.PhotoBsh.values.astype(float)
        self.data.PhotoCsh = self.data.PhotoCsh.values.astype(float)
        self.data.PhotoDsh = self.data.PhotoDsh.values.astype(float)

        self.data.PhotoA = self.data.PhotoA.values.astype(float)
        self.data.PhotoB = self.data.PhotoB.values.astype(float)
        self.data.PhotoC = self.data.PhotoC.values.astype(float)
        self.data.PhotoD = self.data.PhotoD.values.astype(float)
        
    def normalizeToIntegrationTime(self):
        ##### normalize to integration time
        self.data.PhotoAsh = self.data.PhotoAsh.values.astype(float)/self.data.GateShArr.values.astype(float)
        self.data.PhotoA = self.data.PhotoA.values.astype(float)/self.data.GateLgArr.values.astype(float)

        self.data.PhotoBsh = self.data.PhotoBsh.values.astype(float)/self.data.GateShArr.values.astype(float)
        self.data.PhotoB = self.data.PhotoB.values.astype(float)/self.data.GateLgArr.values.astype(float)

        self.data.PhotoCsh = self.data.PhotoCsh.values.astype(float)/self.data.GateShArr.values.astype(float)
        self.data.PhotoC = self.data.PhotoC.values.astype(float)/self.data.GateLgArr.values.astype(float)

        self.data.PhotoDsh = self.data.PhotoDsh.values.astype(float)/self.data.GateShArr.values.astype(float)
        self.data.PhotoD = self.data.PhotoD.values.astype(float)/self.data.GateLgArr.values.astype(float)



    def set_dateTime(self, millisscale = 10):
        self.data.Seconds *= (millisscale/1000.)
        self.data.index = self.data.Seconds

        self.data.Month = np.floor(self.data.MonthDay/100.)
        _simplefill(self.data.Month)

        self.data.Day = self.data.MonthDay - self.data.Month*100
        _simplefill(self.data.Day)
        self.data.Month = self.data.Month.astype(int).apply(lambda x: '{0:0>2}'.format(x))
        self.data.Day = self.data.Day.astype(int).apply(lambda x: '{0:0>2}'.format(x))
        self.GPSHr_P_3 = self.data.GPSHr.copy()
        self.Month_P_1 = self.data.Month.copy()
        self.Day_P_1 = self.data.Day.copy()
        ## time from GPS
        # get rid of stepfunktion
#         GPSunique = df.GPSReadSeconds.dropna().unique()
#         for e,i in enumerate(GPSunique):
#             where = np.where(df.GPSReadSeconds == i)[0][1:]
#             self.data.GPSReadSeconds.values[where] = np.nan

        GPSunique = self.data.GPSHr.dropna().unique()
        for e,i in enumerate(GPSunique):
            where = np.where(self.data.GPSHr == i)[0][1:]
            self.data.GPSHr.values[where] = np.nan

        self.GPSHr_P_1 = self.data.GPSHr.copy()
        # extrapolate and interpolate the time    
        _extrapolate(self.data.index, self.data.GPSHr)
        self.data.GPSHr.interpolate(method='index', inplace= True)
        self.data.GPSHr.dropna(inplace=True)
        self.GPSHr_P_2 = self.data.GPSHr.copy()
        self.data.GPSHr = self.data.GPSHr.apply(lambda x: '%02i:%02i:%09.6f'%(x,60 * (x % 1), 60* ((60 * (x % 1)) %1)))

        ###### DateTime!!
        dateTime = year + '-' +  self.data.Month + '-' + self.data.Day +' ' + self.data.GPSHr
        self.data.index = pd.Series(pd.to_datetime(dateTime, format="%Y-%m-%d %H:%M:%S.%f"), name='Time_UTC')

        self.data = self.data[pd.notnull(self.data.index)]  # gets rid of NaT
    
    def recover_negative_values(self):
        """ this is most likely not working!"""
        _recover_Negatives(self.data.PhotoA, verbose=self.verbose)
        _recover_Negatives(self.data.PhotoB, verbose=self.verbose)
        _recover_Negatives(self.data.PhotoC, verbose=self.verbose)
        _recover_Negatives(self.data.PhotoD, verbose=self.verbose)
        _recover_Negatives(self.data.PhotoAsh, verbose=self.verbose)
        _recover_Negatives(self.data.PhotoBsh, verbose=self.verbose)
        _recover_Negatives(self.data.PhotoCsh, verbose=self.verbose)
        _recover_Negatives(self.data.PhotoDsh, verbose=self.verbose)
    
    def remove_unused_columns(self):
        self.data.drop('var1', axis=1, inplace= True)
        self.data.drop('var2', axis=1, inplace= True)
        self.data.drop('var3', axis=1, inplace= True)
        self.data.drop('time', axis=1, inplace= True)
        self.data.drop('GPSHr', axis=1, inplace= True)
        self.data.drop('MonthDay', axis=1, inplace= True)
        self.data.drop('Month', axis=1, inplace= True)
        self.data.drop('Day', axis=1, inplace= True)
        self.data.drop('GPSReadSeconds', axis=1, inplace= True)
        self.data.drop('GateLgArr', axis=1, inplace= True)
        self.data.drop('GateShArr', axis=1, inplace= True)