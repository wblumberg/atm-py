import numpy as _np
from scipy.optimize import fsolve as _fsolve
from scipy.optimize import curve_fit as _curve_fit
import pandas as _pd
import atmPy.general.timeseries as _timeseries
import warnings as _warnings





def kappa_simple(k, RH, refractive_index = None, inverse = False):
    """Returns the growth factor as a function of kappa and RH.
    This function is based on the simplified model introduced by Rissler et al. (2006).
    It ignores the curvature effect and is therefore independend of the exact particle diameter.
    Due to this simplification this function is valid only for particles larger than 100 nm
    Patters and Kreidenweis (2007).

    Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter representation of hygroscopic
    growth and cloud condensation nucleus activity, 1961-1971. doi:10.5194/acp-7-1961-2007

    Rissler, J., Vestin, A., Swietlicki, E., Fisch, G., Zhou, J., Artaxo, P., & Andreae, M. O. (2005).
    Size distribution and hygroscopic properties of aerosol particles from dry-season biomass burning
    in Amazonia. Atmospheric Chemistry and Physics Discussions, 5(5), 8149-8207. doi:10.5194/acpd-5-8149-2005

    latex expression: $\sqrt[3]{1 + \kappa \cdot \frac{RH}{100 - RH}}$

    Arguments
    ---------
    k: float
        kappa value between 0 (no hygroscopicity) and 1.4 (NaCl; most hygroscopic
        material found in atmospheric
        aerosols)
        If inverse is True than this is the growth factor.
    RH: float
        Relative humidity -> between 0 and 100 (i don't think this simplified
        version will be correct at higher RH?!?)
    inverse: bool.
        if function is inversed, which means instead of the growth factor is
        calculated as a function of the kappa value, it is the other way around.
    Returns
    -------
    float: The growth factor of the particle.
    if refractive_index is given a further float is returned which gives the new refractiv index of the grown particle
    """

    if not inverse:
        if _np.any(k > 1.4) or _np.any(k < 0):
            # txt = '''The kappa value has to be between 0 and 1.4.'''
            # txt = txt + '\nk[k>1.4]:\n%s'%(k[k>1.4]) +'\nk[k<0]:\n%s'%(k[k<0])
            # print(txt)
            # raise ValueError(txt)
            if _np.any(k > 1.4):
                _warnings.warn('There are kappa values lareger than 1.4 in this dataset!')
            if _np.any(k < 0):
                _warnings.warn('There are kappa values smaller than 0 in this dataset!')

    if _np.any(RH > 100) or _np.any(RH < 0):
        txt = """RH has to be between 0 and 100"""
        raise ValueError(txt)

    if inverse:
        kappa = lambda gf,RH: (gf**3 - 1) / (RH / (100 - RH))
        out = kappa(k,RH)
    else:
        growths_factor = lambda k,RH: (1 + (k * (RH/(100 - RH))))**(1/3.)
        out = growths_factor(k,RH)

    # adjust index of refraction
    if not inverse:
        if _np.any(refractive_index):
            nw = 1.33
            # n_mix = lambda n,gf: (n + ((gf-1)*nw))/gf
            n_mix = lambda n, gf: (n + (nw * (gf ** 3 - 1))) / gf ** 3 # This is the correct function for volume mixing ratio
            if type(refractive_index).__name__ == 'DataFrame':
                refractive_index = refractive_index.iloc[:,0]
            # import pdb
            # pdb.set_trace()
            return out, n_mix(refractive_index, out)

    return out


def kappa_from_fofrh_and_sizedist(f_of_RH, dist, wavelength, RH, verbose = False, f_of_RH_collumn = None):
    """
    Calculates kappa from f of RH and a size distribution.
    Parameters
    ----------
    f_of_RH: TimeSeries
    dist: SizeDist
    wavelength: float
    RH: float
        Relative humidity at which the f_of_RH is taken.
    column: string
        when f_of_RH has more than one collumn name the one to be used
    verbose: bool

    Returns
    -------
    TimeSeries
    """

    def minimize_this(gf, sr, f_rh_soll, ext, wavelength, verbose = False):
        gf = float(gf)
        sr_g = sr.apply_growth(gf, how='shift_bins')
        sr_g_opt = sr_g.calculate_optical_properties(wavelength)
        ext_g = sr_g_opt.extinction_coeff_sum_along_d.data.values[0][0]
        f_RH = ext_g / ext
        out = f_RH - f_rh_soll

        if verbose:
            print('test growhtfactor: %s'%gf)
            print('f of RH soll/is: %s/%s'%(f_rh_soll,f_RH))
            print('diviation from soll: %s'%out)
            print('---------')
        return out

    # make sure f_of_RH has only one collumn
    if f_of_RH.data.shape[1] > 1:
        if not f_of_RH_collumn:
            txt = 'f_of_RH has multiple collumns (%s). Please name the one you want to use by setting the f_of_RH_collumn argument.'%(f_of_RH.data.columns)
            raise ValueError(txt)
        else:
            f_of_RH = f_of_RH._del_all_columns_but(f_of_RH_collumn)

    n_values = dist.data.shape[0]
    gf_calc = _np.zeros(n_values)
    kappa_calc = _np.zeros(n_values)
    f_of_RH_aligned = f_of_RH.align_to(dist)
    for e in range(n_values):
        frhsoll = f_of_RH_aligned.data.values[e][0]
        if _np.isnan(frhsoll):
            kappa_calc[e] = _np.nan
            gf_calc[e]  = _np.nan
            continue

        if type(dist.index_of_refraction).__name__ == 'float':
            ior = dist.index_of_refraction
        else:
            ior = dist.index_of_refraction.iloc[e][0]
        if _np.isnan(ior):
            kappa_calc[e] = _np.nan
            gf_calc[e]  = _np.nan
            continue

        sr = dist.copy()
        sr.data = sr.data.iloc[[e],:]
        sr.index_of_refraction = ior
        sr_opt = sr.calculate_optical_properties(wavelength)
        ext = sr_opt.extinction_coeff_sum_along_d.data.values[0][0]


        if ext == 0:
            kappa_calc[e] = _np.nan
            gf_calc[e]  = _np.nan
            continue

        if verbose:
            print('goal for f_rh: %s'%frhsoll)
            print('=======')

        gf_out = _fsolve(minimize_this, 1, args = (sr, frhsoll, ext, wavelength, verbose), factor=0.5, xtol = 0.005)
        gf_calc[e] = gf_out

        if verbose:
            print('resulting gf: %s'%gf_out)
            print('=======\n')

        kappa_calc[e] = kappa_simple(gf_out, RH, inverse=True)
    ts_kappa = _timeseries.TimeSeries(_pd.DataFrame(kappa_calc, index = f_of_RH_aligned.data.index, columns= ['kappa']))
    ts_kappa._data_period = f_of_RH_aligned._data_period
    ts_kappa._y_label = '$\kappa$'

    ts_gf = _timeseries.TimeSeries(_pd.DataFrame(gf_calc, index = f_of_RH_aligned.data.index, columns= ['growth factor']))
    ts_kappa._data_period = f_of_RH_aligned._data_period
    ts_gf._y_label = 'growth factor$'
    return ts_kappa, ts_gf

#########################
##### f of RH

def f_RH_kappa(RH, k, RH0 = 0):
    f_RH = (1 + (k * (RH/(100 - RH)))) / (1 + (k * (RH0/(100 - RH0))))
    return f_RH

def f_RH_gamma(RH, g, RH0 = 0):
    f_RH = ((1 - (RH / 100))**(-g)) / ((1 - (RH0 / 100))**(-g))
    return f_RH


def fofRH_from_dry_wet_scattering(scatt_dry, scatt_wet,RH_dry, RH_wet, data_period = 60, return_fits = False, verbose = False):
    """
    This function was originally written for ARM's AOS dry wet nephelometer proceedure. Programming will likely be needed to make it work for something else.

    Notes
    -----
    For each RH scan in the wet nephelometer an experimental f_RH curve is created by deviding
    scatt_wet by scatt_dry. This curve is then fit by a gamma as well as a kappa parametrizaton.
    Here the dry nephelometer is NOT considered as RH = 0 but its actuall RH (averaged over the
    time of the scann) is considered. I was hoping that this will eliminated a correlation between
    "the ratio" and the dry nephelometer's RH ... it didn't :-(

    Parameters
    ----------
    scatt_dry:  TimeSeries
    scatt_wet:  TimeSeries
    RH_dry:     TimeSeries
    RH_wet:     TimeSeries
    data_period: int,float
        measurement frequency. Might only be needed if return_fits = True ... double check
    return_fits: bool
        If the not just the fit results but also the corresponding curves are going to be returned

    Returns
    -------
    pandas.DataFrame containing the fit results
    pandas.DataFrame containing the fit curves (only if retun_fits == True)

    """

    # some modification to the kappa function so it can be used in the fit routine later
    f_RH_kappa_RH0 = lambda RH0: (lambda RH, k: f_RH_kappa(RH, k, RH0))
    f_RH_gamma_RH0 = lambda RH0: (lambda RH, g: f_RH_gamma(RH, g, RH0))
    # crate the f(RH)/f(RH0)

    #     scatt_dry = _timeseries._timeseries(_pd.DataFrame(scatt_dry))
    #     scatt_dry._data_period = data_period
    #     scatt_wet = _timeseries._timeseries(_pd.DataFrame(scatt_wet))
    #     scatt_wet._data_period = data_period

    f_RH = scatt_wet / scatt_dry
    f_RH.data.columns = ['f_RH']
    #     f_RH = _timeseries._timeseries(_pd.DataFrame(f_RH, columns=['f_RH']))
    #     f_RH._data_period = data_period

    # get start time for next RH-ramp (it always start a few minutes after the full hour)
    start, end = f_RH.get_timespan()
    start_first_section = _np.datetime64('{:d}-{:02d}-{:02d} {:02d}:00:00'.format(start.year, start.month, start.day, start.hour))

    if (start.minute + start.second + start.microsecond) > 0:
        start_first_section += _np.timedelta64(1, 'h')

    # select one hour starting with time defined above. Also align/merge dry and wet RH to it
    i = -1
    fit_res_list = []
    results = _pd.DataFrame(columns=['kappa',
                                     'kappa_std',
                                     'f_RH_85_kappa',
                                     'f_RH_85_kappa_std',
                                     #                                   'f_RH_85_kappa_errp',
                                     #                                   'f_RH_85_kappa_errm',
                                     'gamma',
                                     'gamma_std',
                                     'f_RH_85_gamma',
                                     'f_RH_85_gamma_std',
                                     'wet_neph_max',
                                     'dry_neph_mean',
                                     'dry_neph_std',
                                     'wet_neph_min'], dtype = _np.float64)
    while i < 30:
        i += 1
        # stop if section end is later than end of file
        section_start = start_first_section + _np.timedelta64(i, 'h')
        section_end = start_first_section + _np.timedelta64(i, 'h') + _np.timedelta64(45, 'm')

        if (end - section_end) < _np.timedelta64(0, 's'):
            break

        if verbose:
            print('================')
            print('start of section: ', section_start)
            print('end of section: ', section_end)


        try:
            section = f_RH.zoom_time(section_start, section_end)
        except IndexError:
            if verbose:
                print('section has no data in it!')
            results.loc[section_start] = _np.nan
            continue



        df = section.data.copy().dropna()
        if df.shape[0] < 2:
            if verbose:
                print('no data in section.dropna()!')
            results.loc[section_start] = _np.nan
            continue

        #         section = section.merge(out.RH_nephelometer._del_all_columns_but('RH_NephVol_Wet'))
        #         section = section.merge(out.RH_nephelometer._del_all_columns_but('RH_NephVol_Dry')).data

        section = section.merge(RH_wet)
        section = section.merge(RH_dry).data

        # this is needed to get the best parameterization
        dry_neph_mean = section.RH_NephVol_Dry.mean()
        dry_neph_std = section.RH_NephVol_Dry.std()
        wet_neph_min = section.RH_NephVol_Wet.min()
        wet_neph_max = section.RH_NephVol_Wet.max()
        # clean up
        section.dropna(inplace=True)
        section = section[section.f_RH != _np.inf]
        section = section[section.f_RH != -_np.inf]
        timestamps = section.index.copy()
        section.index = section.RH_NephVol_Wet
        section.drop('RH_NephVol_Wet', axis=1, inplace=True)
        section.drop('RH_NephVol_Dry', axis=1, inplace=True)

        # fitting!!
        if dry_neph_mean > wet_neph_max:
            if verbose:
                print('dry_neph_mean > wet_neph_max!!! something wrong with dry neph!!')
            results.loc[section_start] = _np.nan
            continue

        try:
            kappa, [k_varience] = _curve_fit(f_RH_kappa_RH0(dry_neph_mean), section.index.values, section.f_RH.values)
            # gamma, [varience] = curve_fit(gamma_paramterization, section.index.values, section.f_RH.values)
            gamma, [varience] = _curve_fit(f_RH_gamma_RH0(dry_neph_mean), section.index.values, section.f_RH.values)
        except:
            import pdb
            pdb.set_trace()

        frame_this = {'kappa': kappa[0],
                      'kappa_std': _np.sqrt(k_varience[0]),
                      'f_RH_85_kappa': f_RH_kappa(85, kappa[0]),
                      'f_RH_85_kappa_std': - f_RH_kappa(85, kappa[0]) + f_RH_kappa(85, kappa[0] + _np.sqrt(k_varience[0])),
                      #         'f_RH_85_kappa_errp': f_RH_kappa(85, kappa[0] + _np.sqrt(k_varience[0])),
                      #         'f_RH_85_kappa_errm': f_RH_kappa(85, kappa[0] - _np.sqrt(k_varience[0])),
                      'gamma': gamma[0],
                      'gamma_std': _np.sqrt(varience[0]),
                      'f_RH_85_gamma': f_RH_gamma(85, gamma[0]),
                      'f_RH_85_gamma_std': - f_RH_gamma(85, gamma[0]) + f_RH_gamma(85, gamma[0] + _np.sqrt(varience[0])),
                      'dry_neph_mean': dry_neph_mean,
                      'dry_neph_std': dry_neph_std,
                      'wet_neph_min': wet_neph_min,
                      'wet_neph_max': wet_neph_max}

        results.loc[section_start] = frame_this

        if return_fits:
            # plotting preparation
            RH = section.index.values
            #     RH = _np.linspace(0,100,20)
            fit = f_RH_gamma_RH0(dry_neph_mean)(RH, gamma)
            fit_std_p = f_RH_gamma_RH0(dry_neph_mean)(RH, gamma + _np.sqrt(varience))
            fit_std_m = f_RH_gamma_RH0(dry_neph_mean)(RH, gamma - _np.sqrt(varience))

            fit_k = f_RH_kappa_RH0(dry_neph_mean)(RH, kappa)
            fit_k_std_p = f_RH_kappa_RH0(dry_neph_mean)(RH, kappa + _np.sqrt(k_varience))
            fit_k_std_m = f_RH_kappa_RH0(dry_neph_mean)(RH, kappa - _np.sqrt(k_varience))

            df['fit_gamma'] = _pd.Series(fit, index=df.index)
            df['fit_gamma_stdp'] = _pd.Series(fit_std_p, index=df.index)
            df['fit_gamma_stdm'] = _pd.Series(fit_std_m, index=df.index)

            df['fit_kappa'] = _pd.Series(fit_k, index=df.index)
            df['fit_kappa_stdp'] = _pd.Series(fit_k_std_p, index=df.index)
            df['fit_kappa_stdm'] = _pd.Series(fit_k_std_m, index=df.index)
            fit_res_list.append(df)

    if results.shape[0] == 0:
        results.loc[start] = _np.nan
    results = _timeseries.TimeSeries(results)
    results._data_period = 3600

    if return_fits:
        fit_res = _pd.concat(fit_res_list).sort_index()
        ts = _timeseries.TimeSeries(fit_res)
        ts._data_period = data_period
        fit_res = ts.close_gaps()
        return results, fit_res
    else:

        return results