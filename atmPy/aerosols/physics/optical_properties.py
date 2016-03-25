from copy import deepcopy as _deepcopy

import numpy as np
import pandas as pd
from scipy import integrate

from atmPy.aerosols.size_distribution import \
    sizedist_moment_conversion as _sizedist_moment_conversion
from atmPy.general import timeseries
from atmPy.general import vertical_profile
from atmPy.radiation.mie_scattering import bhmie
import warnings as _warnings


# Todo: Docstring is wrong
# todo: This function can be sped up by breaking it apart. Then have OpticalProperties
#       have properties that call the subfunction on demand
def size_dist2optical_properties(sd, wavelength, n, aod=False, noOfAngles=100):
    """
    !!!Tis Docstring need fixn
    Calculates the extinction crossection, AOD, phase function, and asymmetry Parameter for each layer.
    plotting the layer and diameter dependent extinction coefficient gives you an idea what dominates the overall AOD.

    Parameters
    ----------
    wavelength: float.
        wavelength of the scattered light, unit: nm
    n: float.
        Index of refraction of the scattering particles

    noOfAngles: int, optional.
        Number of scattering angles to be calculated. This mostly effects calculations which depend on the phase
        function.

    Returns
    -------
    OpticalProperty instance

    """
    out = {}
    sdls = sd.convert2numberconcentration()
    index = sdls.data.index
    dist_class = type(sdls).__name__

    if dist_class not in ['SizeDist','SizeDist_TS']:
        raise TypeError('this distribution class (%s) can not be converted into optical property yet!'%dist_class)

    # determin if index of refraction changes or if it is constant
    if isinstance(n, pd.DataFrame):
        n_multi = True
    else:
        n_multi = False
    if not n_multi:
        mie, angular_scatt_func = _perform_Miecalculations(np.array(sdls.bincenters / 1000.), wavelength / 1000., n,
                                                       noOfAngles=noOfAngles)

    if aod:
        #todo: use function that does a the interpolation instead of the sum?!? I guess this can lead to errors when layers are very thick, since centers are used instea dof edges?
        AOD_layer = np.zeros((len(sdls.layercenters)))

    extCoeffPerLayer = np.zeros((len(sdls.data.index.values), len(sdls.bincenters)))
    angular_scatt_func_effective = pd.DataFrame()
    asymmetry_parameter_LS = np.zeros((len(sdls.data.index.values)))

    for i, lc in enumerate(sdls.data.index.values):
        laydata = sdls.data.iloc[i].values # picking a size distribution (either a layer or a point in time)

        if n_multi:
            mie, angular_scatt_func = _perform_Miecalculations(np.array(sdls.bincenters / 1000.), wavelength / 1000., n.iloc[i].values[0],
                                                       noOfAngles=noOfAngles)
        extinction_coefficient = _get_coefficients(mie.extinction_crossection, laydata)

        if aod:
            layerThickness = sdls.layerbounderies[i][1] - sdls.layerbounderies[i][0]
            AOD_perBin = extinction_coefficient * layerThickness
            AOD_layer[i] = AOD_perBin.values.sum()

        extCoeffPerLayer[i] = extinction_coefficient

        scattering_cross_eff = laydata * mie.scattering_crossection

        pfe = (laydata * angular_scatt_func).sum(axis=1)  # sum of all angular_scattering_intensities

        x_2p = pfe.index.values
        y_2p = pfe.values

        # limit to [0,pi]
        y_1p = y_2p[x_2p < np.pi]
        x_1p = x_2p[x_2p < np.pi]

        y_phase_func = y_1p * 4 * np.pi / scattering_cross_eff.sum()
        asymmetry_parameter_LS[i] = .5 * integrate.simps(np.cos(x_1p) * y_phase_func * np.sin(x_1p), x_1p)
        angular_scatt_func_effective[
            lc] = pfe * 1e-12 * 1e6  # equivalent to extCoeffPerLayer # similar to  _get_coefficients (converts everthing to meter)

    if aod:
        out['AOD'] = AOD_layer[~ np.isnan(AOD_layer)].sum()
        out['AOD_layer'] = pd.DataFrame(AOD_layer, index=sdls.layercenters, columns=['AOD per Layer'])
        out['AOD_cum'] = out['AOD_layer'].iloc[::-1].cumsum().iloc[::-1]

    extCoeff_perrow_perbin = pd.DataFrame(extCoeffPerLayer, index=index, columns=sdls.data.columns)

    # if dist_class == 'SizeDist_TS':
    #     out['extCoeff_perrow_perbin'] = timeseries.TimeSeries_2D(extCoeff_perrow_perbin)
    if dist_class == 'SizeDist':
        out['extCoeff_perrow_perbin'] = timeseries.TimeSeries(extCoeff_perrow_perbin)
    else:
        out['extCoeff_perrow_perbin'] = extCoeff_perrow_perbin
    # extCoeff_perrow = pd.DataFrame(extCoeff_perrow_perbin.sum(axis=1), columns=['ext_coeff'])
    # if index.dtype == '<M8[ns]':
    #     out['extCoeff_perrow'] = timeseries.TimeSeries(extCoeff_perrow)
    # else:
    #     out['extCoeff_perrow'] = extCoeff_perrow

    out['parent_type'] = dist_class
    out['asymmetry_param'] = pd.DataFrame(asymmetry_parameter_LS, index=index,
                                          columns=['asymmetry_param'])
    # out['asymmetry_param_alt'] = pd.DataFrame(asymmetry_parameter_LS_alt, index=sdls.layercenters, columns = ['asymmetry_param_alt'])
    # out['OptPropInstance']= OpticalProperties(out, self.bins)
    out['wavelength'] = wavelength
    out['index_of_refraction'] = n
    out['bin_centers'] = sdls.bincenters
    out['bins'] = sdls.bins
    out['binwidth'] = sdls.binwidth
    out['distType'] = sdls.distributionType
    out['angular_scatt_func'] = angular_scatt_func_effective
    # opt_properties = OpticalProperties(out, self.bins)
    # opt_properties.wavelength = wavelength
    # opt_properties.index_of_refractio = n
    # opt_properties.angular_scatt_func = angular_scatt_func_effective  # This is the formaer phase_fct, but since it is the angular scattering intensity, i changed the name
    # opt_properties.parent_dist_LS = self
    if dist_class == 'SizeDist_TS':
        return OpticalProperties_TS(out,parent = sd)


    return out


def hemispheric_backscattering(osf_df):
    """scattering into backwards hemisphere from angulare scattering intensity

    Parameters
    ----------
    osf_df: pandas DataFrame
        This contains the angulare scattering intensity with column names giving the
        angles in radiant


    Returns
    -------
    pandas data frame with the scattering intensities
    """
    def ang_scat_funk2bs(index,ol):
        x = index
        f = ol
        # my phase function goes all the way to two py
        f = f[ x < np.pi]
        x = x[ x < np.pi]
        f_b = f[x >= np.pi/2.]
        x_b = x[x >= np.pi/2.]

        res_b = 2* np.pi * integrate.simps(f_b * np.sin(x_b),x_b)
        return res_b

    bs = np.zeros(osf_df.shape[0])
    index = osf_df.columns
    for i in range(osf_df.shape[0]):
        ol = osf_df.iloc[i,:].values
        bs[i] = ang_scat_funk2bs(index,ol)
    bs = pd.DataFrame(bs, index = osf_df.index)
    return bs

def hemispheric_forwardscattering(osf_df):
    """scattering into forward hemisphere from angulare scattering intensity

    Parameters
    ----------
    osf_df: pandas DataFrame
        This contains the angulare scattering intensity with column names giving the
        angles in radiant

    Returns
    -------
    pandas data frame with the scattering intensities
    """

    def ang_scat_funk2fs(index,ol):
        x = index #ol.index.values
        f = ol

        # my phase function goes all the way to two py
        f = f[ x < np.pi]
        x = x[ x < np.pi]
        f_f = f[x < np.pi/2.]
        x_f = x[x < np.pi/2.]

        res_f = 2* np.pi * integrate.simps(f_f * np.sin(x_f),x_f)
        return res_f

    fs = np.zeros(osf_df.shape[0])
    index = osf_df.columns
    for i in range(osf_df.shape[0]):
        ol = osf_df.iloc[i,:].values
        fs[i] = ang_scat_funk2fs(index,ol)
    fs = pd.DataFrame(fs, index = osf_df.index)
    return fs








#Todo: bins are redundand
# Todo: some functions should be switched of
# todo: right now this for layer and time series, not ok
class OpticalProperties(object):
    def __init__(self, data, parent = None):
        self.parent_sizedist = parent

        self.data_orig = data
        self.wavelength =  data['wavelength']
        self.index_of_refraction = data['index_of_refraction']
        self.extinction_coeff_per_bin = data['extCoeff_perrow_perbin']
        self.angular_scatt_func = data['angular_scatt_func']

        # self.asymmetry_param = data['asymmetry_param']

        # self.angular_scatt_function = data['angular_scatt_func']

        # self.bin_centers = data['bin_centers']

        self.extinction_coeff_sum_along_d = None
        self.mean_effective_diameter = None
        self._parent_type = data['parent_type']
        self.bins = data['bins']
        self.binwidth = data['binwidth']
        self.distributionType = data['distType']
        self._data_period = self.parent_sizedist._data_period


    # @property
    # def mean_effective_diameter(self):
    #     if not self.__mean_effective_diameter:


    # todo: remove
    @property
    def extinction_coeff_sum_along_d(self):
        _warnings.warn('extinction_coeff_sum_along_d is deprecated and will be removed in future versions. Use extingction_coeff instead')
        if not np.any(self.__extinction_coeff_sum_along_d):
            data = self.extinction_coeff_per_bin.data.sum(axis = 1)
            df = pd.DataFrame()
            df['ext_coeff_m^1'] = data
            if self._parent_type == 'SizeDist_TS':
                self.__extinction_coeff_sum_along_d = timeseries.TimeSeries(df)
            elif self._parent_type == 'SizeDist':
                self.__extinction_coeff_sum_along_d = df
            else:
                raise TypeError('not possible for this distribution type')
            self.__extinction_coeff_sum_along_d._data_period = self._data_period
        return self.__extinction_coeff_sum_along_d

    # todo: remove
    @extinction_coeff_sum_along_d.setter
    def extinction_coeff_sum_along_d(self, data):
        self.__extinction_coeff_sum_along_d = data

    @property
    def extinction_coeff(self):
        if not np.any(self.__extinction_coeff_sum_along_d):
            data = self.extinction_coeff_per_bin.data.sum(axis=1)
            df = pd.DataFrame()
            df['ext_coeff_m^1'] = data
            if self._parent_type == 'SizeDist_TS':
                self.__extinction_coeff_sum_along_d = timeseries.TimeSeries(df)
            elif self._parent_type == 'SizeDist':
                self.__extinction_coeff_sum_along_d = df
            else:
                raise TypeError('not possible for this distribution type')
            self.__extinction_coeff_sum_along_d._data_period = self._data_period
        return self.__extinction_coeff_sum_along_d

    @extinction_coeff.setter
    def extinction_coeff(self, data):
        self.__extinction_coeff_sum_along_d = data

    @property
    def hemispheric_backscattering(self):
        if not self.__hemispheric_backscattering:
            self.__hemispheric_backscattering = hemispheric_backscattering(self.angular_scatt_func)
        return self.__hemispheric_backscattering

    @property
    def hemispheric_forwardscattering(self):
        if not self.__hemispheric_forwardscattering:
            self.__hemispheric_forwardscattering = hemispheric_forwardscattering(self.angular_scatt_func)
        return self.__hemispheric_forwardscattering

    @property
    def hemispheric_backscattering_ratio(self):
        if not self.__hemispheric_backscattering_ratio:
            self.__hemispheric_backscattering_ratio = self.hemispheric_backscattering / self.extinction_coeff
        return self.__hemispheric_backscattering_ratio

    @property
    def hemispheric_forwardscattering_ratio(self):
        if not self.hemispheric_forwardscattering_ratio:
            self.__hemispheric_forwardscattering_ratio = self.hemispheric_forwardscattering / self.extinction_coeff
        return self.__hemispheric_forwardscattering_ratio

    def convert_between_moments(self, moment, verbose = False):
        return _sizedist_moment_conversion.convert(self,moment, verbose = verbose)

    def copy(self):
        return _deepcopy(self)



class OpticalProperties_TS(OpticalProperties):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extinction_coeff_per_bin = timeseries.TimeSeries_2D(self.extinction_coeff_per_bin)
        self.extinction_coeff_per_bin._data_period = self.parent_sizedist._data_period

        self.angular_scatt_func = timeseries.TimeSeries_2D(self.angular_scatt_func.transpose())
        self.angular_scatt_func._data_period = self.parent_sizedist._data_period

        self.__hemispheric_forwardscattering = None
        self.__hemispheric_backscattering = None
        self.__hemispheric_backscattering_ratio = None
        self.__hemispheric_forwardscattering_ratio = None


    @property
    def hemispheric_backscattering(self):
        if not self.__hemispheric_backscattering:
            out = hemispheric_backscattering(self.angular_scatt_func.data)
            out = timeseries.TimeSeries(out)
            out._data_period = self.angular_scatt_func._data_period
            self.__hemispheric_backscattering = out
        return self.__hemispheric_backscattering

    @hemispheric_backscattering.setter
    def hemispheric_backscattering(self,value):
        self.__hemispheric_backscattering = value

    @property
    def hemispheric_forwardscattering(self):
        if not self.__hemispheric_forwardscattering:
            out = hemispheric_forwardscattering(self.angular_scatt_func.data)
            out = timeseries.TimeSeries(out)
            out._data_period = self.angular_scatt_func._data_period
            self.__hemispheric_forwardscattering = out
        return self.__hemispheric_forwardscattering


    @hemispheric_forwardscattering.setter
    def hemispheric_forwardscattering(self, value):
        self.__hemispheric_forwardscattering = value

    @property
    def hemispheric_backscattering_ratio(self):
        """ratio between backscattering and overall scattering"""
        if not self.__hemispheric_backscattering_ratio:
            self.__hemispheric_backscattering_ratio = self.hemispheric_backscattering / self.extinction_coeff
        return self.__hemispheric_backscattering_ratio

    @property
    def hemispheric_forwardscattering_ratio(self):
        """ratio between forwardscattering and over scattering"""
        if not self.__hemispheric_forwardscattering_ratio:
            self.__hemispheric_forwardscattering_ratio = self.hemispheric_forwardscattering / self.extinction_coeff
        return self.__hemispheric_forwardscattering_ratio



#Todo: bins are redundand
# Todo: some functions should be switched of
# todo: right now this for layer and time series, not ok
class Old_OpticalProperties(object):
    def __init__(self, data, bins):
        # self.data = data['extCoeffPerLayer']
        # self.data = data['extCoeff_perrow_perbin']
        self.data_orig = data
        #todo: the following is just a quick fix, the structure is kind of messy
        if 'AOD' in data.keys():
            self.AOD = data['AOD']
        self.bins = bins
        self.layercenters = self.data.index.values
        self.asymmetry_parameter_LS = data['asymmetry_param']
        # self.asymmetry_parameter_LS_alt = data['asymmetry_param_alt']

        # ToDo: to define a distribution type does not really make sence ... just to make the stolen plot function happy
        self.distributionType = 'dNdlogDp'

    #todo: what is the nead for that?
    def get_extinction_coeff_verticle_profile(self):
        """
        Creates a verticle profile of the extinction coefficient.
        """
        ext = self.data.sum(axis=1)
        ext = pd.DataFrame(ext, columns=['ext. coeff.'])
        ext.index.name = 'Altitude'
        out = ExtinctionCoeffVerticlProfile(ext, self, self.wavelength, self.index_of_refractio)
        # out.wavelength = self.wavelength
        # out.n = self.index_of_refractio
        # out.parent = self
        return out

#     def plot_AOD_cum(self, color=plt_tools.color_cycle[0], linewidth=2, ax=None, label='cumulative AOD',
#                      extra_info=True):
#         if not ax:
#             f,a = plt.subplots()
#         else:
#             a = ax
#         # a = self.data_orig['AOD_cum'].plot(color=color, linewidth=linewidth, ax=ax, label=label)
#         g, = a.plot(self.data_orig['AOD_cum']['AOD per Layer'], self.data_orig['AOD_cum'].index, color=color, linewidth=linewidth, label=label)
#
#         # g = a.get_lines()[-1]
#         g.set_label(label)
#         a.legend()
#         # a.set_xlim(0, 3000)
#         a.set_ylabel('Altitude (m)')
#         a.set_xlabel('AOD')
#         txt = '''$\lambda = %s$ nm
# n = %s
# AOD = %.4f''' % (self.data_orig['wavelength'], self.data_orig['n'], self.data_orig['AOD'])
#         if extra_info:
#             a.text(0.7, 0.7, txt, transform=a.transAxes)
#         return a

    # def _getXYZ(self):
    #     out = SizeDist_LS._getXYZ(self)
    #     return out
    #
    # def plot_extCoeffPerLayer(self,
    #                           vmax=None,
    #                           vmin=None,
    #                           scale='linear',
    #                           show_minor_tickLabels=True,
    #                           removeTickLabels=['500', '700', '800', '900'],
    #                           plotOnTheseAxes=False, cmap=plt_tools.get_colorMap_intensity(),
    #                           fit_pos=True,
    #                           ax=None):
    #     f, a, pc, cb = SizeDist_LS.plot(self,
    #                                     vmax=vmax,
    #                                     vmin=vmin,
    #                                     scale=scale,
    #                                     show_minor_tickLabels=show_minor_tickLabels,
    #                                     removeTickLabels=removeTickLabels,
    #                                     plotOnTheseAxes=plotOnTheseAxes,
    #                                     cmap=cmap,
    #                                     fit_pos=fit_pos,
    #                                     ax=ax)
    #     cb.set_label('Extinction coefficient ($m^{-1}$)')

        return f, a, pc, cb

class ExtinctionCoeffVerticlProfile(vertical_profile.VerticalProfile):
    def __init__(self, ext, parent, wavelength, index_of_refraction):
        super(ExtinctionCoeffVerticlProfile, self).__init__(ext)
        self.parent = parent
        self.wavelength = wavelength
        self.index_of_refraction = index_of_refraction

    def plot(self, *args, **kwargs):
        a = super(ExtinctionCoeffVerticlProfile, self).plot(*args, **kwargs)
        a.set_xlabel('Extinction coefficient (m$^{-1}$)')
        return a


def _perform_Miecalculations(diam, wavelength, n, noOfAngles=100.):
    """
    Performs Mie calculations

    Parameters
    ----------
    diam:       NumPy array of floats
                Array of diameters over which to perform Mie calculations; units are um
    wavelength: float
                Wavelength of light in um for which to perform calculations
    n:          complex
                Ensemble complex index of refraction

    Returns
        panda DataTable with the diameters as the index and the mie_scattering results in the different collumns
        total_extinction_coefficient: this takes the sum of all particles crossections of the particular diameter in a qubic
                                      meter. This is in principle the AOD of an L

    """


    diam = np.asarray(diam)

    extinction_efficiency = np.zeros(diam.shape)
    scattering_efficiency = np.zeros(diam.shape)
    absorption_efficiency = np.zeros(diam.shape)

    extinction_crossection = np.zeros(diam.shape)
    scattering_crossection = np.zeros(diam.shape)
    absorption_crossection = np.zeros(diam.shape)

    # phase_function_natural = pd.DataFrame()
    angular_scattering_natural = pd.DataFrame()
    # extinction_coefficient = np.zeros(diam.shape)
    # scattering_coefficient = np.zeros(diam.shape)
    # absorption_coefficient = np.zeros(diam.shape)



    # Function for calculating the size parameter for wavelength l and radius r
    sp = lambda r, l: 2. * np.pi * r / l
    for e, d in enumerate(diam):
        radius = d / 2.

        # print('sp(radius, wavelength)', sp(radius, wavelength))
        # print('n', n)
        # print('d', d)

        mie = bhmie.bhmie_hagen(sp(radius, wavelength), n, noOfAngles, diameter=d)
        values = mie.return_Values_as_dict()
        extinction_efficiency[e] = values['extinction_efficiency']

        # print("values['extinction_crosssection']",values['extinction_crosssection'])


        scattering_efficiency[e] = values['scattering_efficiency']
        absorption_efficiency[e] = values['extinction_efficiency'] - values['scattering_efficiency']

        extinction_crossection[e] = values['extinction_crosssection']
        scattering_crossection[e] = values['scattering_crosssection']
        absorption_crossection[e] = values['extinction_crosssection'] - values['scattering_crosssection']

        # phase_function_natural[d] = values['phaseFct_natural']['Phase_function_natural'].values
        angular_scattering_natural[d] = mie.get_angular_scatt_func().natural.values

        # print('\n')

    # phase_function_natural.index = values['phaseFct_natural'].index
    angular_scattering_natural.index = mie.get_angular_scatt_func().index

    out = pd.DataFrame(index=diam)
    out['extinction_efficiency'] = pd.Series(extinction_efficiency, index=diam)
    out['scattering_efficiency'] = pd.Series(scattering_efficiency, index=diam)
    out['absorption_efficiency'] = pd.Series(absorption_efficiency, index=diam)

    out['extinction_crossection'] = pd.Series(extinction_crossection, index=diam)
    out['scattering_crossection'] = pd.Series(scattering_crossection, index=diam)
    out['absorption_crossection'] = pd.Series(absorption_crossection, index=diam)
    return out, angular_scattering_natural


def _get_coefficients(crossection, cn):
    """
    Calculates the extinction, scattering or absorbtion coefficient

    Parameters
    ----------
    crosssection:   float
                    Units are um^2
    cn:             float
                    Particle concentration in cc^-1

    Returns
    --------
    coefficient in m^-1.  This is the differential AOD.
    """
    crossection = crossection.copy()
    cn = cn.copy()
    crossection *= 1e-12  # conversion from um^2 to m^2
    cn *= 1e6  # conversion from cm^-3 to m^-3
    coefficient = cn * crossection

    # print('cn',cn)
    # print('crossection', crossection)
    # print('coeff',coefficient)
    # print('\n')

    return coefficient

def vertical_profile2accumulative_AOD(timeseries):
    data = timeseries.data.copy()
    data.dropna(inplace = True)
    accu_aod = np.zeros(data.shape)
    for col,k in enumerate(data.keys()):
        series = data[k]
        # series.dropna(inplace = True)
        y = series.values*1e-6
        x = series.index.values

        x = x[::-1]
        y = y[::-1]


        st = 0
        # for e,i in enumerate(x):
        for e in range(x.shape[0]):
            end = e+1
            accu_aod[e][col] = -integrate.simps(y[st:end],x[st:end])

    accu_aod = pd.DataFrame(accu_aod, index = x, columns=data.keys())
    accu_aod = vertical_profile.VerticalProfile(accu_aod)

    accu_aod._x_label = 'AOD$_{abs}$'
    return accu_aod