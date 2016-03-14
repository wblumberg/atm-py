__author__ = 'htelg'

import numpy as _np
from scipy import stats as _stats
import matplotlib.pylab as _plt
from atmPy.tools import plt_tools as _plt_tools

def find_closest(array, value, how = 'closest'):
    """Finds the element of an array which is the closest to a given number and returns its index

    Arguments
    ---------
    array:    array
        The array to search thru.
    value:    float or array-like.
        Number (list of numbers) to search for.
    how: string
        'closest': look for the closest value
        'closest_low': look for the closest value that is smaller than value
        'closest_high': look for the closest value that is larger than value

    Return
    ------
    integer or array
        position of closest value(s)"""

    if _np.any(_np.isnan(array)) or _np.any(_np.isnan(value)):
        txt = '''Array or value contains nan values; that will not work'''
        raise ValueError(txt)

    if type(value).__name__ in ('float', 'int', 'float64', 'int64'):
        single = True
        value = _np.array([value], dtype=float)

    elif type(value).__name__ in ('list', 'ndarray'):
        single = False
        pass

    else:
        raise ValueError('float,int,array or list are ok types for value. You provided %s' % (type(value).__name__))

    out = _np.zeros((len(value)), dtype=int)
    for e, i in enumerate(value):
        nar = array - i
        if how == 'closest':
            pass
        elif how == 'closest_low':
            nar[nar > 0] = array.max()
        elif how == 'closest_high':
            nar[nar < 0] = array.max()
        else:
            txt = 'The keyword argument how has to be one of the following: "closest", "closest_low", "closest_high"'
            raise ValueError(txt)
        out[e] = _np.abs(nar).argmin()
    if single:
        out = out[0]
    return out


class Correlation(object):
    def __init__(self, data, correlant, remove_zeros = True, index = False):
        """This object is for testing correlation in two two data sets.

        Parameters
        ----------
        data and correlant: 1D arry
            These are the two data set which are compared
        remove_zeros: bool
            If zeros ought to be deleted. Datasets often contain zeros that are the
            result of invalid data. If there is the danger that this introduces a
            bias set it to False"""

        data = data.copy()
        correlant = correlant.copy()
        self.__pearson_r = None
        self.__linear_regression = None
        self.__linear_regression_function = None
        if remove_zeros:
            correlant = correlant[data != 0]
            if type(index) != bool:
                index = index[data != 0]
            data = data[data != 0]

            data = data[correlant != 0]
            if type(index) != bool:
                index = index[correlant != 0]
            correlant = correlant[correlant != 0]

        self._data = data
        self._correlant = correlant
        self._index = index
        self._x_label_correlation = 'Data'
        self._y_label_correlation = 'Correlant'
        self._x_label_orig = 'Item'
        self._y_label_orig_data = 'Data'
        self._y_label_orig_correlant = 'Correlant'

    @property
    def pearson_r(self):
        if not self.__pearson_r:
            self.__pearson_r = _stats.pearsonr(self._data, self._correlant)
        return self.__pearson_r

    @property
    def linear_regression(self):
        if not self.__linear_regression:
            self.__linear_regression = _stats.linregress(self._data, self._correlant)
        return self.__linear_regression

    @property
    def linear_regression_function(self):
        if not self.__linear_regression_function:
            self.__linear_regression_function = lambda x: x * self.linear_regression.slope + self.linear_regression.intercept
        return self.__linear_regression_function

    def plot_pearson(self, gridsize = 100, cm = 'auto', p_value = True, colorbar = False, ax = None, **kwargs):
        if not ax:
            f,a = _plt.subplots()
        else:
            f = ax.get_figure()
            a = ax

        # cm = plt_tools.()
        # cm = plt.cm.gist_earth_r
        # cm = plt.cm.hot_r
#         cm = plt.cm.Blues

        a.set_xlabel(self._x_label_correlation)
        a.set_ylabel(self._y_label_correlation)

        if cm == 'auto':
            cm = _plt.cm.copper_r

        cm.set_under('w')

        hb = a.hexbin(self._data, self._correlant, gridsize=gridsize, cmap=cm, vmin = 0.001, **kwargs)
        if colorbar:
            f.colorbar(hb, ax = a)
#         linreg_func = lambda x: x * linreg.slope + linreg.intercept
        # data.min()

        x_reg_func = _np.array([self._data.min(), self._data.max()])
        y_reg_func = self.linear_regression_function(x_reg_func)

        color = _plt_tools.color_cycle[2]
        a.plot(x_reg_func, y_reg_func, lw = 2, color = color)


        txt = '$r = %0.2f$'%(self.pearson_r[0])
        if p_value:
            txt += '\n$p = %0.2f$'%(self.pearson_r[1])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        a.text(0.1,0.9, txt, transform=a.transAxes, horizontalalignment='left', verticalalignment='top', bbox = props)
        return a

    def plot_original_data(self, ax = None, **kwargs):
        if not ax:
            f,a = _plt.subplots()
        else:
            f = ax.get_figure()
            a = ax

        a.set_xlabel(self._x_label_orig)

        if type(self._index) != bool:
            a.plot(self._index, self._data, linewidth = 2, color = _plt_tools.color_cycle[0], **kwargs)
        else:
            a.plot(self._data, linewidth = 2, color = _plt_tools.color_cycle[0], **kwargs)

        a.set_ylabel(self._y_label_orig_data)

        a.tick_params(axis = 'y', left = True, color = _plt_tools.color_cycle[0], zorder = 99)
        a.spines['left'].set_color(_plt_tools.color_cycle[0])
        a.spines['left'].set_zorder(99)

        a2 = a.twinx()
        if type(self._index) != bool:
            a2.plot(self._index,self._correlant, linewidth = 2, color = _plt_tools.color_cycle[1])
        else:
            a2.plot(self._correlant, linewidth = 2, color = _plt_tools.color_cycle[1])

        a2.set_ylabel(self._y_label_orig_correlant)

        a2.tick_params(axis = 'y', right = True, color = _plt_tools.color_cycle[1])
        a2.spines['right'].set_color(_plt_tools.color_cycle[1])
        a2.spines['left'].set_visible(False)


        if type(self._index).__name__ == 'DatetimeIndex':
            f.autofmt_xdate()
        return a, a2

    def plot_pearsonANDoriginal_data(self, gridsize = 20, cm = _plt.cm.Blues, p_value = True, width_ratio = [1.5, 2], corr_kwargs = {}, orig_kwargs = {}):
        f, (a_corr, a_orig) = _plt.subplots(1,2, gridspec_kw = {'width_ratios':width_ratio})
        f.set_figwidth(f.get_figwidth()*1.7)
        a1 = self.plot_pearson(gridsize=gridsize, cm = cm, p_value=p_value, ax = a_corr, **corr_kwargs)
        a2,a3 = self.plot_original_data(ax = a_orig, **orig_kwargs)
        return a1, a2, a3
