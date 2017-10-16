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

    def zoom_time(self, start=None, end=None, copy=True):
        kazrnew = self.copy()
        kazrnew.reflectivity = self.reflectivity.zoom_time(start=start, end=end, copy=copy)
        kazrnew.signal2noise_ratio = self.signal2noise_ratio.zoom_time(start=start, end=end, copy=copy)
        return kazrnew

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
