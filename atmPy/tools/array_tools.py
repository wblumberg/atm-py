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
    def __init__(self, data, correlant, remove_zeros = True):
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
            data = data[data != 0]
            data = data[correlant != 0]
            correlant = correlant[correlant != 0]

        self._data = data
        self._correlant = correlant

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

    def plot_pearson(self, gridsize = 100, cm = _plt.cm.Blues, p_value = True):
        # cm = plt_tools.()
        # cm = plt.cm.gist_earth_r
        # cm = plt.cm.hot_r
#         cm = plt.cm.Blues
        f,a = _plt.subplots()
        a.hexbin(self._data, self._correlant, gridsize=gridsize, cmap=cm)

#         linreg_func = lambda x: x * linreg.slope + linreg.intercept
        # data.min()

        x_reg_func = _np.array([self._data.min(), self._data.max()])
        y_reg_func = self.linear_regression_function(x_reg_func)

        color = _plt_tools.color_cycle[1]
        a.plot(x_reg_func, y_reg_func, lw = 2, color = color)


        txt = '$r = %0.2f$'%(self.pearson_r[0])
        if p_value:
            txt += '\n$p = %0.2f$'%(self.pearson_r[1])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        a.text(0.1,0.9, txt, transform=a.transAxes, horizontalalignment='left', verticalalignment='top', bbox = props)
        return a

    def plot_original_data(self):
        f,a = _plt.subplots()
        a.set_xlabel('arbitrary')

        a.plot(self._data, linewidth = 2, color = _plt_tools.color_cycle[0])
        a.set_ylabel('data')

        a2 = a.twinx()
        a2.plot(self._correlant, linewidth = 2, color = _plt_tools.color_cycle[1])
        a2.set_ylabel('correlant')
        return a, a2