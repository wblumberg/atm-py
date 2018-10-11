import atmPy.general.timeseries as _timeseries
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm as _LogNorm
import numpy as _np
from copy import deepcopy as _deepcopy

class Reflectivity(_timeseries.TimeSeries_2D):
    def __init__(self, *args, parent= None, **kwargs):
        super().__init__(*args ,**kwargs)
        self._parent = parent

    def plot(self, snr_max = None, norm = 'linear', **kwargs):
        if 'pc_kwargs' in kwargs:
            pc_kwargs = kwargs['pc_kwargs']
        else:
            pc_kwargs = {}

        if 'cmap' not in pc_kwargs:
            pc_kwargs['cmap'] = plt.cm.gist_gray_r
        if 'norm' not in pc_kwargs:
            if norm == 'log':
                print(norm)
                pc_kwargs['norm'] = _LogNorm()
            #         if 'vmin' not in pc_kwargs:
            #             pc_kwargs['vmin'] = vmin

        kwargs['pc_kwargs'] = pc_kwargs

        if snr_max:
            refl = self.copy()
            refl.data[self._parent.signal2noise_ratio.data < snr_max] = _np.nan
            out = refl.plot(norm = norm, **kwargs)
        else:
            out = super().plot(**kwargs)
        return out

class Kazr(object):
    def __init__(self):
        self._reflectivity = None
        self._signal2noise_ratio = None

    def average_time(self, window):
        """
        Averages each of the relevant properties. See timeseries.TimeSeries.average_time for details.
        Parameters
        ----------
        window: tuple
            e.g. (1,'m')

        Returns
        -------
        Kazr instances with changes applied
        """

        kzr = self.copy()
        kzr.reflectivity = kzr.reflectivity.average_time(window)
        return kzr


    def zoom_time(self, start=None, end=None, copy=True):
        kazrnew = self.copy()
        kazrnew.reflectivity = self.reflectivity.zoom_time(start=start, end=end, copy=copy)
        kazrnew.signal2noise_ratio = self.signal2noise_ratio.zoom_time(start=start, end=end, copy=copy)
        return kazrnew

    def discriminate_by_signal2noise_ratio(self, minimu_snr):
        """I know there is that kwarg in the plot function which allows me to do this. This was necessary in order to
        average over time and still be able to discriminate through the snr. After averaging over time the snr is
        useless.

        Parameters
        ----------
        minimu_snr: float
            All values of reflectivity where the snr is smaller then that value are set to nan.

        Returns
        -------
        Kazr instance with changes applied
        """

        kzr = self.copy()
        kzr.reflectivity.data[self.signal2noise_ratio.data < minimu_snr] = _np.nan
        return  kzr

    @property
    def reflectivity(self):
        return self._reflectivity

    @reflectivity.setter
    def reflectivity(self, value, **kwargs):
        if type(value).__name__ == 'Reflectivity':
            self._reflectivity = value
        else:
            self._reflectivity = Reflectivity(value, parent = self, **kwargs)

    @property
    def signal2noise_ratio(self):
        return self._signal2noise_ratio

    @signal2noise_ratio.setter
    def signal2noise_ratio(self, value, **kwargs):
        if type(value).__name__ == 'TimeSeries_2D':
            self._signal2noise_ratio = value
        else:
            self._signal2noise_ratio = _timeseries.TimeSeries_2D(value, **kwargs)

    def copy(self):
        return _deepcopy(self)
