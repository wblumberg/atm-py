__author__ = 'htelg'

import numpy as _np
from scipy import stats as _stats
import matplotlib.pylab as _plt
from atmPy.tools import plt_tools as _plt_tools
import scipy.odr as _odr
import warnings as _warnings

_colors = _plt.rcParams['axes.prop_cycle'].by_key()['color']

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


def reverse_binary(variable, no_bits):
    """This converts all numbers into binary of length no_bits. Then it reverses the
    binaries and finally converts it into integer again.
    This is usefull for quality flags that are often represented in integers of
    which each position of the corresponding binary tells you something about a
    different qualty criteria. Sometimes bad values are at the beginning sometimes
    at the end and reversing is desired.

    Parameters
    ==========
    variable: ndarray or pandas object

    Returns
    =======
    what ever you put in

    Examples
    ========
    >>> a = np.array([1,0,0,2,0,8])
    >>> array_tools.reverse_binary(a,4)
    array([8, 0, 0, 4, 0, 1])
    """
    variable = variable.copy()
    rep = '{0:0%sb}'%no_bits
    func = _np.vectorize(lambda i: int(rep.format(i)[::-1],2))
    variable[:] = func(variable)
    return variable


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = _np.array(values)
    quantiles = _np.array(quantiles)
    if sample_weight is None:
        sample_weight = _np.ones(len(values))
    sample_weight = _np.array(sample_weight)
    assert _np.all(quantiles >= 0) and _np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = _np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = _np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with _np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= _np.sum(sample_weight)
    return _np.interp(quantiles, weighted_quantiles, values)

class Correlation(object):
    def __init__(self, data, correlant, remove_zeros = True, index = False, odr_function = 'linear',
                 # sx = 1, sy = 1,
                 weights = 'scaled',
                 poly_order = 1
                 ):
        """This object is for testing correlation in two two data sets.

        Parameters
        ----------
        data and correlant: 1D arry
            These are the two data set which are compared
        remove_zeros: bool
            If zeros ought to be deleted. Datasets often contain zeros that are the
            result of invalid data. If there is the danger that this introduces a
            bias set it to False
        odr_function: string
            currently 'linear' only (odr only)
        weights: 'str' ([scaled], constant)
            odr only!
            scaled: the weights will scale with the x-data. Since we usually assume uncertainties to be relative to
                    the data rather then absolute this is the standard setting
            constant:   bsically this means not weighted ... in principle you could apply different for x and y ... not
                        implemented yet
        poly_order: int [1]
            odr only
            for the orthogonal distance regression (odr) the polynomial function is used. poly_order gives the order of
            the polynomial. Default is 1 which results in a linear regression.


        Deprecated
        ----------
        sx, sy: float
            covarience estimates (odr only)"""

        data = data.copy()
        correlant = correlant.copy()
        self.__pearson_r = None
        self.__linear_regression = None
        self.__linear_regression_function = None
        self.__linear_regression_zero = None
        self.__linear_regression_zero_function= None
        self.__orthogonal_distance_regression = None

        if remove_zeros:
            correlant = correlant[data != 0]
            if type(index) != bool:
                index = index[data != 0]
            data = data[data != 0]

            data = data[correlant != 0]
            if type(index) != bool:
                index = index[correlant != 0]
            correlant = correlant[correlant != 0]

        # nans have to be removed
        correlant = correlant[~ _np.isnan(data)]
        if type(index) != bool:
            index = index[~ _np.isnan(data)]
        data = data[~ _np.isnan(data)]

        data = data[~ _np.isnan(correlant)]
        if type(index) != bool:
            index = index[~ _np.isnan(correlant)]
        correlant = correlant[~ _np.isnan(correlant)]

        self._data = data
        self._correlant = correlant
        self._maxvalue = max([data.max(),correlant.max()])
        self._minvalue = min([data.min(), correlant.min()])
        self._index = index
        self.odr_function = odr_function
        self.weights = weights
        self.poly_order = poly_order
        # self.sy = sy
        # self.sx = sx

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
    def linear_regression_zero_intersect(self):
        if not self.__linear_regression_zero:
            x = self._data
            x = x[:, _np.newaxis]
            self.__linear_regression_zero = _np.linalg.lstsq(x, self._correlant)
        return self.__linear_regression_zero

    @property
    def linear_regression_function(self):
        if not self.__linear_regression_function:
            self.__linear_regression_function = lambda x: x * self.linear_regression.slope + self.linear_regression.intercept
        return self.__linear_regression_function

    @property
    def linear_regression_zero_intersect_function(self):
        if not self.__linear_regression_zero_function:
            self.__linear_regression_zero_function = lambda x: x * self.linear_regression_zero_intersect[0]
        return self.__linear_regression_zero_function

    @property
    def orthogonla_distance_regression(self):
        if not self.__orthogonal_distance_regression:
            if self.weights == 'scaled':
                std = 0.1
                mydata = _odr.RealData(self._data, self._correlant, sx=self._data * std,
                                         sy=self._correlant * std)  # , wd=1. / self.sx ** 2, we=1. / self.sy ** 2)
            elif self.weights == 'constant':
                mydata = _odr.Data(self._data, self._correlant, wd=1. / self.sx ** 2, we=1. / self.sy ** 2)
            else:
                raise ValueError('weights has to be scalar or constant')
            if self.odr_function!= 'linear':
                raise('only "linear" allowed at this point, programming required!')
            model = _odr.polynomial(self.poly_order)
            myodr = _odr.ODR(mydata, model)
            myoutput = myodr.run()
            self.__orthogonal_distance_regression = {'model': myodr,
                                                     'output': myoutput,
                                                     'function': _np.polynomial.Polynomial(myoutput.beta)
#lambda x: _odr.unilinear.fcn(myoutput.beta, x)
                                                     }
        return self.__orthogonal_distance_regression

    def plot_regression(self, reg_type = 'odr',zero_intersect=False, gridsize=100, cm='auto', xlim=None,
                        ylim=None, colorbar=False, ax=None, aspect='equal',
                        fit_res_kwargs = {},
                        show_slope_1 = True,
                        # 'pos':(0.1, 0.9),
                        #                   'show_params': ['r','r2','p','m', 'c', 's']
                        #                   },
                        # vmin=0.001,
                        hexbin_kwargs = {}, plot_kwargs = {}):
        """

        Parameters
        ----------
        reg_type: string
            which type of regression:
                - 'simple'
                - 'odr' -- orthogonlal distance regression (scipy.odr)
        gridsize:
        cm: matplotlib.color map
        xlim: int or float
            upper limit of x. Similar to set_xlim(right = ...) in addition it
            adjusts the gridsize so hexagons are not getting streched
        ylim: int or float
            as xlim just for y-axis
        show_slope_1: bool [True]
            if the 1:1 line is to be shown
        p_value: bool
            if the p-value is given in the text box
        colorbar: bool
        ax: bool or matplotlib.Axes instance
            If desired to plot on another axes.
        fit_res_kwargs: dict ... allowed keys
            pos: tuple of len=2
            show_params: ['r','r2','p','m', 'c', 's']
                list of fit-parameters to show
            bb_fc: color of box face color
            bb_ec: color of box edge color
            bb_lw: line width of box
        Returns
        -------

        """
        if not ax:
            f, a = _plt.subplots()
        else:
            f = ax.get_figure()
            a = ax

        if aspect == 'auto':
            ratio = 14 / 20  # at this ratio hexagons look symmetric at the particular setting
        elif aspect == 'equal':
            pass
        else:
            ratio = aspect * 20 / 14

        a.set_xlabel(self._x_label_correlation)
        a.set_ylabel(self._y_label_correlation)

        if cm == 'auto':
            cm = _plt.cm.gnuplot
            cm.set_under([1,1,1,0])
            hexbin_kwargs['cmap'] = cm

        if type(gridsize) == tuple:
            gridsize_x, gridsize_y = gridsize
        else:
            if xlim:
                if type(xlim).__name__ in ['int', 'float']:
                    # xratio = self._data.max() / xlim
                    # gridsize_x = int(gridsize * xratio)
                    xmax = xlim
                    xmin = self._data.min()
                    xlim = (xmin, xmax)
                elif type(xlim).__name__ == 'tuple''':
                    xmax = xlim[1]
                    xmin = xlim[0]
            else:
                xmax = self._maxvalue
                xmin = self._minvalue

            xratio = (self._data.max() - self._data.min()) / (xmax - xmin)
            gridsize_x = int(gridsize * xratio)
            # else:
            #     gridsize_x = gridsize

            if ylim:
                if type(ylim).__name__ in ['int', 'float']:
                    # yratio = self._correlant.max() / ylim
                    # gridsize_y = int(ratio * gridsize * yratio)
                    ymax = ylim
                    ymin = self._correlant.min()
                    ylim = (ymin, ymax)
                elif type(ylim).__name__ == 'tuple''':
                    ymax = ylim[1]
                    ymin = ylim[0]
            else:
                ymax = self._maxvalue
                ymin = self._minvalue
            yratio = (self._correlant.max() - self._correlant.min()) / (ymax - ymin)
            gridsize_y = int(gridsize * yratio)
        # else:
        #     if type(gridsize) != tuple:
        #         gridsize_y = int(gridsize * ratio)

        if type(gridsize) == tuple:
            gridsize_new = gridsize
        else:
            gridsize_new = (gridsize_x, gridsize_y)

        # import pdb
        # pdb.set_trace()
        hb = a.hexbin(self._data, self._correlant, gridsize=gridsize_new, **hexbin_kwargs)
        hb.set_clim(0.01) # this is so every empty bin results to get the color defined in cm.set_under() -> transparent

        if colorbar:
            f.colorbar(hb, ax=a)
            #         linreg_func = lambda x: x * linreg.slope + linreg.intercept
        # data.min()

        if 'color' not in plot_kwargs.keys():
            plot_kwargs['color'] =  _colors[2]
        if 'simple' in reg_type:
            x_reg_func = _np.array([self._data.min(), self._data.max()])
            if zero_intersect:
                y_reg_func = self.linear_regression_zero_intersect_function(x_reg_func)
                slope = self.linear_regression_zero_intersect[0]
                intersect = 0
                std = (self._correlant - self.linear_regression_zero_intersect_function(self._data)).std()
            else:
                y_reg_func = self.linear_regression_function(x_reg_func)
                slope = self.linear_regression.slope
                intersect = self.linear_regression.intercept
                # std = self.linear_regression.stderr
                std = (self._correlant - self.linear_regression_function(self._data)).std()
            a.plot(x_reg_func, y_reg_func, **plot_kwargs)

        if 'odr' in reg_type:
            x_reg_func = self._data #_np.array([self._data.min(), self._data.max()])
            y_reg_func = self.orthogonla_distance_regression['function'](x_reg_func)
            slope = self.orthogonla_distance_regression['output'].beta[1]
            intersect = self.orthogonla_distance_regression['output'].beta[0]
            std = _np.sqrt(self.orthogonla_distance_regression['output'].res_var)
            a.plot(x_reg_func, y_reg_func, label = 'odr fit', **plot_kwargs)
            # a.set_xlim((self._data.min(), self._data.max()))
            # a.set_ylim((self._correlant.min(), self._correlant.max()))

        # color = _plt_tools.color_cycle[2]

        if type(fit_res_kwargs) == dict:
            txt_r = '$r = %0.2f$' % (self.pearson_r[0])
            txt_r2= '$r^2 = %0.2f$' % ((self.pearson_r[0]) ** 2)
            # if p_value:
            txt_p= '$p = %0.2f$' % (self.pearson_r[1])
            txt_m= '$m = %0.2f$' % (slope)
            txt_c= '$c = %0.2f$' % (intersect)
            txt_s= '$s = %0.2f$' % (std)

            txtl = []

            if 'show_params' not in fit_res_kwargs.keys():
                fit_res_kwargs['show_params'] = ['r', 'r2', 'p', 'm', 'c', 's']

            if 'pos' not in fit_res_kwargs.keys():
                fit_res_kwargs['pos'] = (0.1, 0.9)

            if 'bb_fc' not in fit_res_kwargs.keys():
                fit_res_kwargs['bb_fc'] = [1,1,1,0.5]

            if 'bb_ec' not in fit_res_kwargs.keys():
                fit_res_kwargs['bb_ec'] = [0,0,0,1]

            if 'bb_lw' not in fit_res_kwargs.keys():
                fit_res_kwargs['bb_lw'] = _plt.rcParams['axes.linewidth']

            for fr in fit_res_kwargs['show_params']:
                if fr == 'r':
                    txtl.append(txt_r)
                elif fr == 'r2':
                    txtl.append(txt_r2)
                elif fr == 'p':
                    txtl.append(txt_p)
                elif fr == 'm':
                    txtl.append(txt_m)
                elif fr == 'c':
                    txtl.append(txt_c)
                elif fr == 's':
                    txtl.append(txt_s)
                else:
                    raise

            txt = '\n'.join(txtl)



            props = dict(boxstyle='round',
                         facecolor=fit_res_kwargs['bb_fc'],
                         edgecolor = fit_res_kwargs['bb_ec'],
                         lw = fit_res_kwargs['bb_lw']
                         )
            pos = fit_res_kwargs['pos']
            a.text(pos[0], pos[1], txt, transform=a.transAxes, horizontalalignment='left', verticalalignment='top', bbox=props)

        if xlim:
            a.set_xlim(xlim)
        else:
            a.set_xlim(self._minvalue, self._maxvalue)

        if ylim:
            a.set_ylim(ylim)
        else:
            a.set_ylim(self._minvalue, self._maxvalue)

        if show_slope_1:
            a.plot([self._minvalue, self._maxvalue], [self._minvalue, self._maxvalue], ls='--', color = _colors[1], label = '1:1')

        if aspect == 'auto':
            pass
        elif aspect == 'equal':
            a.set_aspect('equal')
        else:
            x0, x1 = a.get_xlim()
            y0, y1 = a.get_ylim()
            a.set_aspect(aspect * (abs(x1 - x0) / abs(y1 - y0)))
        a.legend()
        return a, hb

    # todo: allow xlim and ylim to be tuples so you can devine a limit range rather then just the upper limit
    def plot_pearson(self, zero_intersect = False, gridsize = 100, cm = 'auto', xlim = None,
                     ylim = None, colorbar = False, ax = None, aspect = 'auto', fit_res = (0.1,0.9), vmin = 0.001, **kwargs):
        """

        Parameters
        ----------
        gridsize:
        cm: matplotlib.color map
        xlim: int or float
            upper limit of x. Similar to set_xlim(right = ...) in addition it
            adjusts the gridsize so hexagons are not getting streched
        ylim: int or float
            as xlim just for y-axis
        p_value: bool
            if the p-value is given in the text box
        colorbar: bool
        ax: bool or matplotlib.Axes instance
            If desired to plot on another axes.
        kwargs

        Returns
        -------

        """

        _warnings.warn('plot_pearson is deprecated, use plot_regression instead')
        if not ax:
            f,a = _plt.subplots()
        else:
            f = ax.get_figure()
            a = ax

        if aspect == 'auto':
            ratio = 14/20 #at this ratio hexagons look symmetric at the particular setting
        else:
            ratio = aspect * 20/14

        a.set_xlabel(self._x_label_correlation)
        a.set_ylabel(self._y_label_correlation)

        if cm == 'auto':
            cm = _plt.cm.copper_r

        cm.set_under('w')



        if xlim:
            if type(xlim).__name__ in ['int', 'float']:
                # xratio = self._data.max() / xlim
                # gridsize_x = int(gridsize * xratio)
                xmax = xlim
                xmin = self._data.min()
                xlim = (xmin,xmax)
            elif type(xlim).__name__ == 'tuple''':
                xmax = xlim[1]
                xmin = xlim[0]

            xratio = (self._data.max() - self._data.min()) / (xmax - xmin)
            gridsize_x = int(gridsize * xratio)
        else:
            gridsize_x = gridsize

        if ylim:
            if type(ylim).__name__ in ['int', 'float']:
                # yratio = self._correlant.max() / ylim
                # gridsize_y = int(ratio * gridsize * yratio)
                ymax = ylim
                ymin = self._correlant.min()
                ylim = (ymin, ymax)
            elif type(ylim).__name__ == 'tuple''':
                ymax = ylim[1]
                ymin = ylim[0]
            xratio = (self._correlant.max() - self._correlant.min()) / (ymax - ymin)
            gridsize_y = int(gridsize * xratio)
        else:
            gridsize_y = int(gridsize * ratio)


        gridsize_new = (gridsize_x, gridsize_y)

        # import pdb
        # pdb.set_trace()
        hb = a.hexbin(self._data, self._correlant, gridsize=gridsize_new, cmap=cm, vmin = vmin, **kwargs)

        if xlim:
            a.set_xlim(xlim)
        if ylim:
            a.set_ylim(ylim)

        if colorbar:
            f.colorbar(hb, ax = a)
#         linreg_func = lambda x: x * linreg.slope + linreg.intercept
        # data.min()

        x_reg_func = _np.array([self._data.min(), self._data.max()])

        if zero_intersect:
            y_reg_func = self.linear_regression_zero_intersect_function(x_reg_func)
            slope = self.linear_regression_zero_intersect[0]
            intersect = 0
            std = (self._correlant - self.linear_regression_zero_intersect_function(self._data)).std()
        else:
            y_reg_func = self.linear_regression_function(x_reg_func)
            slope = self.linear_regression.slope
            intersect = self.linear_regression.intercept
            # std = self.linear_regression.stderr
            std = (self._correlant - self.linear_regression_function(self._data)).std()

        color = _plt_tools.color_cycle[2]
        if fit_res:
            a.plot(x_reg_func, y_reg_func, lw = 2, color = color)


        txt = '$r = %0.2f$'%(self.pearson_r[0])
        txt += '\n$r^2 = %0.2f$' % ((self.pearson_r[0])**2)
        # if p_value:
        txt += '\n$p = %0.2f$'%(self.pearson_r[1])
        txt += '\n$m = %0.2f$'%(slope)
        txt += '\n$c = %0.2f$'%(intersect)
        txt += '\n$std = %0.2f$'%(std)

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        if fit_res:
            a.text(fit_res[0],fit_res[1], txt, transform=a.transAxes, horizontalalignment='left', verticalalignment='top', bbox = props)

        if aspect != 'auto':
            x0, x1 = a.get_xlim()
            y0, y1 = a.get_ylim()
            a.set_aspect(aspect * (abs(x1 - x0) / abs(y1 - y0)))
        return a

    def plot_original_data(self, ax = None, **kwargs):
        if not ax:
            f,a = _plt.subplots()
        else:
            f = ax.get_figure()
            a = ax

        a.set_xlabel(self._x_label_orig)

        if type(self._index) != bool:
            a.plot(self._index, self._data, color = _plt_tools.color_cycle[0], **kwargs)
        else:
            a.plot(self._data, color = _plt_tools.color_cycle[0], **kwargs)

        g = a.get_lines()[-1]
        g.set_marker('.')

        a.set_ylabel(self._y_label_orig_data)

        a.tick_params(axis = 'y', left = True, color = _plt_tools.color_cycle[0], zorder = 99)
        a.spines['left'].set_color(_plt_tools.color_cycle[0])
        a.spines['left'].set_zorder(99)

        a2 = a.twinx()
        if type(self._index) != bool:
            a2.plot(self._index,self._correlant, linewidth = 2, color = _plt_tools.color_cycle[1], **kwargs)
        else:
            a2.plot(self._correlant, linewidth = 2, color = _plt_tools.color_cycle[1])

        g = a2.get_lines()[-1]
        g.set_marker('.')

        a2.set_ylabel(self._y_label_orig_correlant)

        a2.tick_params(axis = 'y', right = True, color = _plt_tools.color_cycle[1])
        a2.spines['right'].set_color(_plt_tools.color_cycle[1])
        a2.spines['left'].set_visible(False)


        if type(self._index).__name__ == 'DatetimeIndex':
            # f.autofmt_xdate()
            _plt.setp(a.xaxis.get_majorticklabels(), rotation=30 )
        return a, a2

    def plot_pearsonANDoriginal_data(self, gridsize = 20, zero_intersect = False, xlim = None, ylim = None, cm = 'auto', width_ratio = [1.5, 2], corr_kwargs = {}, orig_kwargs = {}):
        f, (a_corr, a_orig) = _plt.subplots(1,2, gridspec_kw = {'width_ratios':width_ratio})
        f.set_figwidth(f.get_figwidth()*1.7)
        a1 = self.plot_pearson(zero_intersect = zero_intersect, gridsize=gridsize, cm = cm, xlim = xlim, ylim = ylim, ax = a_corr, **corr_kwargs)
        a2,a3 = self.plot_original_data(ax = a_orig, **orig_kwargs)
        return a1, a2, a3

    def plot_regressionANDoriginal_data(self, reg_type = 'simple', gridsize = 20, zero_intersect = False, xlim = None, ylim = None, cm = 'auto', width_ratio = [1.5, 2], corr_kwargs = {}, orig_kwargs = {}):
        f, (a_corr, a_orig) = _plt.subplots(1,2, gridspec_kw = {'width_ratios':width_ratio})
        f.set_figwidth(f.get_figwidth()*1.7)
        a1 = self.plot_regression(reg_type = reg_type, zero_intersect = zero_intersect, gridsize=gridsize, cm = cm, xlim = xlim, ylim = ylim, ax = a_corr, **corr_kwargs)
        a2,a3 = self.plot_original_data(ax = a_orig, **orig_kwargs)
        return a1, a2, a3


    def plot_residual(self, gridsize = 100, norm = 'relative', ax = False):
        """Plots the residual from the odr!!! odr only!!

        Parameters
        ----------
        gridsize: int
        norm: str ([relative], absolute)
            if the residual is to be normalized to the data -> relative deviation ... or not
        ax: axes instance to plot on

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        matplotlib.collections.PolyCollection
        """
        if not ax:
            f,a = _plt.subplots()
        else:
            a = ax
            f = ax.get_figure()

        cm = _plt.cm.gnuplot
        cm.set_under([1, 1, 1, 0])
        if norm == 'relative':
            scale = self._data
        elif norm == 'absolute':
            scale = 1
        else:
            raise ValueError('norm has to be absolute or relative')
        pc = a.hexbin(self._data, self.orthogonla_distance_regression['output'].delta / scale, gridsize=gridsize)
        pc.set_clim(0.01)
        pc.set_cmap(cm)
        return a,pc
