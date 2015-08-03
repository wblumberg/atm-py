__author__ = 'htelg'

import pylab as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from geopy.distance import vincenty
import numpy as np
from copy import deepcopy
from atmPy.tools import time_tools


def merge_timeseries(ts_list):
    ts_data_list = [i.data for i in ts_list]
    # try:
    # merged = pd.concat(ts_data_list).sort_index().interpolate().reindex(ts_data_list[0].index)
    # except ValueError:
    #     raise ValueError(
    #         'There is a problem with the time axes. Make sure you limit the data set to a reasonable time interval (e.g. duration of flight)')
    catsortinterp = pd.concat(ts_data_list).sort_index().interpolate()
    merged = catsortinterp.groupby(catsortinterp.index).mean().reindex(ts_data_list[0].index)
    return TimeSeries(merged.iloc[1:-1])


class TimeSeries(object):
    """
    This class simplifies the handling of housekeeping information from measurements.
    Typically this class is created by a housekeeping function of the particular instrument.

    Notes
    -----
    Make sure the passed pandas DataFrame includes the following column names:
     - barometric_pressure
     - altitude

    Attributes
    ----------
    data:  pandas DataFrame with index=DateTime and columns = housekeeping parameters
    """

    def __init__(self, data, info=None):
        self._data = data
        self.info = info

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data


    def copy(self):
        return deepcopy(self)

    def plot_all(self):
        """Plot each parameter separately versus time

        Returns
        -------
        list of matplotlib axes object """

        axes = self.data.plot(subplots=True, figsize=(plt.rcParams['figure.figsize'][0], self.data.shape[1] * 4))
        return axes

    def plot_map(self):
        """Plots a map of the flight path

        Note
        ----
        packages: matplotlib-basemap,
        """
        lon_center = (self.data.Lon.values.max() + self.data.Lon.values.min()) / 2.
        lat_center = (self.data.Lat.values.max() + self.data.Lat.values.min()) / 2.

        points = np.array([self.data.Lat.values, self.data.Lon.values]).transpose()
        distances_from_center_lat = np.zeros(points.shape[0])
        distances_from_center_lon = np.zeros(points.shape[0])
        for e, p in enumerate(points):
            distances_from_center_lat[e] = vincenty(p, (lat_center, p[1])).m
            distances_from_center_lon[e] = vincenty(p, (p[0], lon_center)).m

        lat_radius = distances_from_center_lat.max()
        lon_radius = distances_from_center_lon.max()
        scale = 1
        border = scale * 2 * np.array([lat_radius, lon_radius]).max()

        height = border + lat_radius
        width = border + lon_radius
        bmap = Basemap(projection='aeqd',
                       lat_0=lat_center,
                       lon_0=lon_center,
                       width=width,
                       height=height,
                       resolution='f')

        # Fill the globe with a blue color
        wcal = np.array([161., 190., 255.]) / 255.
        bmap.drawmapboundary(fill_color=wcal)
        # Fill the continents with the land color map.fillcontinents(color=’coral’,lake_color=’aqua’)
        grau = 0.9
        bmap.fillcontinents(color=[grau, grau, grau], lake_color=wcal)
        bmap.drawcoastlines()
        x, y = bmap(self.data.Lon.values, self.data.Lat.values)
        bmap.plot(x, y,
                  color='m')

        return bmap


    def zoom_time(self, start=None, end=None, copy=True):
        """ Selects a strech of time from a housekeeping instance.

        Arguments
        ---------
        start (optional):   string - Timestamp of format '%Y-%m-%d %H:%M:%S.%f' or '%Y-%m-%d %H:%M:%S'
        end (optional):     string ... as start
        copy (optional):    bool - if False the instance will be changed. Else, a copy is returned

        Returns
        -------
        If copy is True:  housekeeping instance
        else:             nothing (instance is changed in place)


        Example
        -------
        >>> from atmPy.instruments.piccolo import piccolo
        >>> launch = '2015-04-19 08:20:22'
        >>> landing = '2015-04-19 10:29:22'
        >>> hk = piccolo.read_file(filename) # create housekeeping instance
        >>> hk_zoom = zoom_time(hk, start = launch, end= landing)
        """

        if copy:
            housek = self.copy()
        else:
            housek = self

        if start:
            start = time_tools.string2timestamp(start)
        if end:
            end = time_tools.string2timestamp(end)

        housek.data = housek.data.truncate(before=start, after=end)
        if copy:
            return housek
        else:
            return


    def plot_versus_pressure_sep_axes(self, what):
        what = self.data[what]

        ax = self.data.barometric_pressure.plot()
        ax.legend()
        ax.set_ylabel('Pressure (hPa)')

        ax2 = ax.twinx()
        what.plot(ax=ax2)
        g = ax2.get_lines()[0]
        g.set_color('red')
        ax2.legend(loc=4)
        return ax, ax2

    def plot_versus_pressure(self, what, ax=False):
        what = self.data[what]

        if ax:
            a = ax
        else:
            f, a = plt.subplots()
        a.plot(self.data.barometric_pressure.values, what)
        a.set_xlabel('Barometric pressure (mbar)')

        return a

    def plot_versus_altitude_sep_axes(self, what):
        what = self.data[what]

        ax = self.data.altitude.plot()
        ax.legend()
        ax.set_ylabel('Altitude (m)')

        ax2 = ax.twinx()
        what.plot(ax=ax2)
        g = ax2.get_lines()[0]
        g.set_color('red')
        ax2.legend(loc=4)
        return ax, ax2

    def plot_versus_altitude(self, what, ax=False, figsize=None):
        """ Plots selected columns versus altitude

        Arguments
        ---------
        what: {'all', key, list of keys}

        Returns
        -------
        matplotlib.axes instance
        """

        allowed = ['altitude', 'Altitude', 'Height']

        if what == 'all':
            what = self.data.keys()

        found = False
        for i in allowed:
            try:
                x = self.data[i]
                found = True
                # print('found %s'%i)
                break
            except KeyError:
                continue

        if not found:
            txt = 'TimeSeries instance has no attribute associated with altitude (%s)' % allowed
            raise AttributeError(txt)

        f, ax = plt.subplots(len(what), sharex=True, gridspec_kw={'hspace': 0.1})

        if not figsize:
            f.set_figheight(4 * len(what))
        else:
            f.set_size_inches(figsize)

        for e, a in enumerate(ax):
            a.plot(x, self.data[what[e]], label=what[e])
            a.legend()

        # a = data[what].plot(subplots = True, figsize = figsize)
        # a[-1].set_xlabel('Altitude (m)')
        # what = self.data[what]
        #
        # if ax:
        # a = ax
        # else:
        #     f, a = plt.subplots()
        # a.plot(self.data.altitude.values, what)
        # a.set_xlabel('Altitude (m)')

        return a

    # def plot_versus_altitude_all(self):
    # axes = []
    #     for key in self.data.keys():
    #         f, a = plt.subplots()
    #         a.plot(self.data.altitude, self.data[key], label=key)
    #         a.legend()
    #         a.grid(True)
    #         axes.append(a)
    #     return axes

    def get_timespan(self):
        """
        Returns the first and last value of the index, which should be the first and last timestamp

        Returns
        -------
        numpy array of datetime64 objects
        """
        return self.data.index.values[[0, -1]]



