__author__ = 'mrichardson, Hagen Telg'

from math import fmod, sin, cos, asin

import ephem
import numpy as np
import pandas as pd

from atmPy.general.constants import a2r, r2a

__julian = {"day": 0., "cent": 0.}


class solar(object):

    def __init__(self, ltime):

        julian = solar.juliandates(ltime)
        self.__jday = julian["day"]
        self.__jcent = julian["cent"]

        self.lon = 0
        self.lat = 0

    sinrad = lambda x: sin(a2r(x))
    cosrad = lambda x: cos(a2r(x))

    def juliandates(self, ltime):
        """
        Calculate a Julian date for a given local time

        Parameters
        ----------
        ltime:      float
                    Local time calculated as seconds since Jan 1 1904
        Returns
        --------
        dictionary
            Returns a dictionary of two floats containing julian day and century.
        """
        # Julian day is the continuous count of days since the beginning of the Julian period.
        self.__jday = ltime/(3600*24)+1462+2415018.5
        self.__jcent = (self.__jday-2451545)/36525
        return None

    def __oblelip(self):

        return ((21.448-self.__jcent*(self.__jcent*
                                     (0.00059-(self.__jcent*0.001813))+46.815))/60+26)/60+23

    def __gemeanlon(self):
        return fmod((self.__jcent*0.0003032+36000.76983)*self.__jcent+280.46646, 360)

    def __meananom(self):
        return self.__jcent*(self.__jcent*0.0001537-35999.05029)+357.52911

    def __eartheccen(self):
        return self.__jcent*(self.__jcent*1.267e-7+4.2037e-5)-0.016708634

    def __centsun(self):

        f = lambda x: sin(a2r(x))

        a = f(3)*0.000289
        b = f(2)*(0.019993-self.__jcent*0.000101)
        c = f(1)*(self.__jcent*(self.__jcent**1.45e-5+0.004817)-1.914602)

        return a+b+c

    def __oblcorr(self):
        return self.cosrad(self.__jcent*1934.136-125.04)*0.00256+self.__oblelip()

    def __truelon(self):
        return self.__gemeanlon() + self.__centsun()

    def __app(self):
        a = self.__truelon()-0.00569
        a -= self.sinrad(self.__jcent*1934.136-125.04)*0.00478
        return a

    def __declang(self):
        return r2a(asin(self.sinrad(self.__oblcorr())*self.sinrad(self.__app())))

    def __eq_time(self):
        return None


def get_sun_position(lat, lon, datetime_UTC, elevation=0):
    """returns elevation and azimuth angle of the sun
    Arguments:
    ----------
    lat, lon: float
        latitude and longitude of the observer (e.g. Denver, lat = 39.7392, lon = -104.9903)
    datetime_UTC: datetime instance or strint ('2015/7/6 19:00:00')
        time of interestes in UTC
    elevation: float, optional.
        elevation of observer.

    Returns
    -------
    tuple of two floats
        elevation and azimuth angle in radians.
    """
    obs = ephem.Observer()
    obs.lat = lat
    obs.long = lon
    obs.elevation = elevation
    # obs.date = '2015/7/6 19:00:00'
    obs.date = datetime_UTC  # datetime.datetime.now() + datetime.timedelta(hours = 6)
    #     print(obs)
    sun = ephem.Sun()
    sun.compute(obs)
    return sun.alt, sun.az


def get_sun_position_TS(timeseries):
    """Returns the position, polar and azimuth angle, of the sun in the sky for a given time and location.

    Arguments
    ---------
    timeseries: pandas.DataFrame instance with the index being of type datetime (e.g. atmPy.timeseries).
        This is typically a housekeeping/telemetry timeseries. It must contain the columns
        Lat, Lon, and Height

    Returns
    -------
    pandas.DataFram with two collums for the elevation and azimuth angle
    Furthermore the timeseries gets two new collumns with the two angles
    """
    lat = timeseries.data.Lat.values.astype(str)
    lon = timeseries.data.Lon.values.astype(str)
    alti = timeseries.data.Altitude.values
    t = timeseries.data.Lat.index
    sunpos = np.zeros((lat.shape[0], 2))
    # sunpos = np.zeros((2,2))
    for e, i in enumerate(lat):
        if 0 == 1:
            break
        sunpos[e] = get_sun_position(lat[e], lon[e], t[e], elevation=alti[e])
    #     return sunpos
    timeseries.data['Solar_position_elevation'] = pd.Series(sunpos[:, 0], index=timeseries.data.index)
    timeseries.data['Solar_position_azimuth'] = pd.Series(sunpos[:, 1], index=timeseries.data.index)
    # return pd.DataFrame(sunpos, columns=['elevation', 'azimuth'], index=timeseries.data.index)
    return timeseries
