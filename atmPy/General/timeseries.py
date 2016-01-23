__author__ = 'htelg'

from copy import deepcopy

import numpy as np
import pandas as pd
import pylab as plt
from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D

from atmPy.radiation import solar
from atmPy.tools import time_tools


# Todo: get rid of this class
# def merge_timeseries(ts_list):
#     """ Merges a list of timeseries into one series. The returned timeseries has the same time-axes as the first
#     timeseries in ts_list. Missing or offset data points are linearly interpolated.
#
#     Argument
#     --------
#     ts_list: list.
#         List of TimeSeries objects.
#
#     Returns
#     -------
#     TimeSeries object
#
#     """
#     warnings.warn("THIS IS OLD, please use the merge attribute of the TimeSeries class")
#     ts_data_list = [i.data for i in ts_list]
#     # try:
#     # merged = pd.concat(ts_data_list).sort_index().interpolate().reindex(ts_data_list[0].index)
#     # except ValueError:
#     #     raise ValueError(
#     #         'There is a problem with the time axes. Make sure you limit the data set to a reasonable time interval (e.g. duration of flight)')
#     catsortinterp = pd.concat(ts_data_list).sort_index().interpolate()
#     merged = catsortinterp.groupby(catsortinterp.index).mean().reindex(ts_data_list[0].index)
#     return TimeSeries(merged.iloc[1:-1])

def load_csv(fname):
    """Loads the dat of a saved timesereis instance and creates a new TimeSeries instance

    Arguments
    ---------
    fname: str.
        Path to the file to load"""
    data = pd.read_csv(fname, index_col=0)
    data.index = pd.to_datetime(data.index)
    return TimeSeries(data)

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

    # Todo: inherit docstring
    def get_sun_position(self):
        """read docstring of solar.get_sun_position_TS"""
        out = solar.get_sun_position_TS(self)
        return out

    def convert2verticalprofile(self):
        hk_tmp = self.copy()
        hk_tmp.data['TimeUTC'] = hk_tmp.data.index
        hk_tmp.data.index = hk_tmp.data.Altitude
        return hk_tmp



    def merge(self, ts):
        """ Merges current with other timeseries. The returned timeseries has the same time-axes as the current
        one (as opposed to the one merged into it). Missing or offset data points are linearly interpolated.

        Argument
        --------
        ts: timeseries or one of its subclasses.
            List of TimeSeries objects.

        Returns
        -------
        TimeSeries object or one of its subclasses

        """
        ts_this = self.copy()
        ts_data_list = [ts_this.data, ts.data]
        catsortinterp = pd.concat(ts_data_list).sort_index().interpolate()
        merged = catsortinterp.groupby(catsortinterp.index).mean().reindex(ts_data_list[0].index)
        ts_this.data = merged
        return ts_this

    def copy(self):
        return deepcopy(self)

    def plot_all(self):
        """Plot each parameter separately versus time

        Returns
        -------
        list of matplotlib axes object """

        axes = self.data.plot(subplots=True, figsize=(plt.rcParams['figure.figsize'][0], self.data.shape[1] * 4))
        return axes

    def plot_map(self, resolution = 'c', three_d=False):
        """Plots a map of the flight path

        Note
        ----
        packages: matplotlib-basemap,

        Arguments
        ---------
        three_d: bool.
            If flight path is plotted in 3D. unfortunately this does not work very well (only costlines)
        """

        data = self.data.copy()
        data = data.loc[:,['Lon','Lat']]
        data = data.dropna()

        lon_center = (data.Lon.values.max() + data.Lon.values.min()) / 2.
        lat_center = (data.Lat.values.max() + data.Lat.values.min()) / 2.

        points = np.array([data.Lat.values, data.Lon.values]).transpose()
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
        if not three_d:
            bmap = Basemap(projection='aeqd',
                           lat_0=lat_center,
                           lon_0=lon_center,
                           width=width,
                           height=height,
                           resolution=resolution)

            # Fill the globe with a blue color
            wcal = np.array([161., 190., 255.]) / 255.
            boundary = bmap.drawmapboundary(fill_color=wcal)

            grau = 0.9
            continents = bmap.fillcontinents(color=[grau, grau, grau], lake_color=wcal)
            costlines = bmap.drawcoastlines()
            x, y = bmap(data.Lon.values, data.Lat.values)
            path = bmap.plot(x, y,
                             color='m')
            return bmap

        else:
            bmap = Basemap(projection='aeqd',
                       lat_0=lat_center,
                       lon_0=lon_center,
                       width=width,
                       height=height,
                       resolution=resolution)

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.add_collection3d(bmap.drawcoastlines())
            x, y = bmap(self.data.Lon.values, self.data.Lat.values)
            # ax.plot(x, y,self.data.Altitude.values,
            #           color='m')
            N = len(x)
            for i in range(N - 1):
                color = plt.cm.jet(i / N)
                ax.plot(x[i:i + 2], y[i:i + 2], self.data.Altitude.values[i:i + 2],
                        color=color)
            return bmap, ax



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
        >>> from atmPy.for_removal.piccolo import piccolo
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

        try:
            housek.data = housek.data.truncate(before=start, after=end)
        except KeyError:
            txt = '''This error is most likely related to the fact that the index of the timeseries is not in order.
                  Run the sort_index() attribute of the DataFrame'''
            raise KeyError(txt)

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
        if ax:
            f = ax[0].get_figure()
        else:
            f, ax = plt.subplots(len(what), sharex=True, gridspec_kw={'hspace': 0.1})
            if len(what) == 1:
                ax = [ax]

        if not figsize:
            f.set_figheight(4 * len(what))
        else:
            f.set_size_inches(figsize)

        for e, a in enumerate(ax):
            a.plot(x, self.data[what[e]], label=what[e])
            a.legend()
            a.set_xlabel('Altitude')

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

        return ax

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
        tuple of timestamps
        """
        start = self.data.index[0]
        end = self.data.index[-1]
        print('start: %s' % start.strftime('%Y-%m-%d %H:%M:%S.%f'))
        print('end:   %s' % end.strftime('%Y-%m-%d %H:%M:%S.%f'))
        return start, end

    def save(self, fname):
        """currently this simply saves the data of the timeseries

        Arguments
        ---------
        fname: str.
            Path to the file."""

        self.data.to_csv(fname)
