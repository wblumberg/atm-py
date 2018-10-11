from atmPy.general import timeseries as _timeseries
import matplotlib.pylab as _plt
from matplotlib.colors import LogNorm as _LogNorm
from copy import deepcopy

class Ceilometer(object):
    def __init__(self):
        self._backscatter = None
        self._cloudbase = None

    @property
    def cloudbase(self):
        return self._cloudbase

    @cloudbase.setter
    def cloudbase(self, value, **kwargs):
        if type(value).__name__ == 'Cloud_base':
            self._cloudbase = value
        else:
            self._cloudbase = Cloud_base(value, **kwargs)

    @property
    def backscatter(self):
        #         if type(self._backscatter).__name__ == 'NoneType':
        return self._backscatter

    @backscatter.setter
    def backscatter(self, value, **kwargs):
        if type(value).__name__ == 'Backscatter':
            self._backscatter = value
        else:
            self._backscatter = Backscatter(value, **kwargs)

    def zoom_time(self, start=None, end=None, copy=True):
        ceilnew = self.copy()
        ceilnew.backscatter = self.backscatter.zoom_time(start=start, end=end, copy=copy)
        ceilnew.cloudbase = self.cloudbase.zoom_time(start=start, end=end, copy=copy)
        return ceilnew

    def copy(self):
        return deepcopy(self)

class Backscatter(_timeseries.TimeSeries_2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, *args, vmin=1e2, norm='log', **kwargs):
        if 'pc_kwargs' in kwargs:
            pc_kwargs = kwargs['pc_kwargs']
            assert (type(pc_kwargs) == dict)
        else:
            pc_kwargs = {}

        if 'cmap' not in pc_kwargs:
            pc_kwargs['cmap'] = _plt.cm.gist_gray_r
        if 'norm' not in pc_kwargs:
            if norm == 'log':
                pc_kwargs['norm'] = _LogNorm()
        if 'vmin' not in pc_kwargs:
            pc_kwargs['vmin'] = vmin

        kwargs['pc_kwargs'] = pc_kwargs

        out = super().plot(*args, **kwargs)
        f,a,lc,cb = out
        a.set_ylabel('Altitude (m)')
        return out


class Cloud_base(_timeseries.TimeSeries):
    def plot(self, which='first', **kwargs):
        if not ('linestyle' in kwargs.keys()) or not ('ls' in kwargs.keys()):
            kwargs['ls'] = ''
        if not 'marker' in kwargs.keys():
            kwargs['marker'] = '_'

        if which == 'first':
            cbs = self._del_all_columns_but('First_cloud_base')
            a = cbs.plot(which='all', **kwargs)
        elif which == 'all':
            a = super().plot(**kwargs)
        else:
            raise ValueError('not implemented yet')

        return a