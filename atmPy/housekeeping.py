__author__ = 'htelg'

import pylab as plt


class HouseKeeping(object):
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

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def plot_all(self):
        """Plot each parameter separately versus time

        Returns
        -------
        list of matplotlib axes object """

        axes = self.data.plot(subplots=True, figsize=(plt.rcParams['figure.figsize'][0], self.data.shape[1] * 4))
        return axes

    def zoom(self, before=None, after=None, axis=None, copy=True):
        """Returns a truncated version of the housekeeping data

        Arguments
        ---------
        see pandas truncate function for details

        Returns
        -------
        HouseKeeping instance

        Example
        -------
        >>> from atmPy.instruments.POPS import housekeeping
        >>> hk = housekeeping.read_housekeeping('19700101_003_POPS_HK.csv')
        >>> hkFlightOnly = hk.zoom(before= '1969-12-31 16:08:55', after = '1969-12-31 18:30:00')
        """

        return HouseKeeping(self.data.truncate(before=before, after=after, axis=axis, copy=copy))

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

    def plot_versus_altitude(self, what, ax=False):
        what = self.data[what]

        if ax:
            a = ax
        else:
            f, a = plt.subplots()
        a.plot(self.data.altitude.values, what)
        a.set_xlabel('Altitude (m)')

        return a

    def plot_versus_altitude_all(self):
        axes = []
        for key in self.data.keys():
            f, a = plt.subplots()
            a.plot(self.data.altitude, self.data[key], label=key)
            a.legend()
            a.grid(True)
            axes.append(a)
        return axes