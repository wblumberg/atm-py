from atmPy.general import timeseries as _timeseries
# from atmPy.data_archives.arm import _tools
import pandas as _pd
from atmPy.data_archives.arm._netCDF import ArmDataset as _ArmDataset
import numpy as _np
from atmPy.tools import decorators
from atmPy.aerosols.physics import hygroscopic_growth as hygrow

import pdb as _pdb


def calculate_f_RH(noaaaos, RH_center, RH_tolerance, which):
    """

    Parameters
    ----------
    noaaaos: noaaaos.ArmDataset instance
    RH_center: int
        The wet nephelometer RH value varies. This is the RH value at which the
        neph data is used. (typical value is 85)
    RH_tolerance: float
        Defines the range of data around RH_center that is included,
        range = RH_center +- RH_tolerance
        Typical value for RH_tolerance is 1
    which: str
        The nephelometer has 3 wavelength channels. Choose between:
        "all", "gree", "red", or "blue".


    Returns
    -------
    TimeSeries instance

    """
    lim = (RH_center - RH_tolerance, RH_center + RH_tolerance)
    rh_wet = noaaaos.RH_nephelometer.data.RH_NephVol_Wet.copy()
    rh_wet.values[rh_wet.values > lim[1]] = _np.nan
    rh_wet.values[rh_wet.values < lim[0]] = _np.nan
# rh_wet.plot()
    if which == 'all':
        which = ['green', 'red', 'blue']
    else:
        which = [which]

    df = _pd.DataFrame(index = noaaaos.scatt_coeff.data.index)
    for col in which:
        if col == 'green':
            wet_column = 'Bs_G_Wet_1um_Neph3W_2'
            dry_column = 'Bs_G_Dry_1um_Neph3W_1'
        elif col == 'red':
            wet_column = 'Bs_R_Wet_1um_Neph3W_2'
            dry_column = 'Bs_R_Dry_1um_Neph3W_1'
        elif col == 'blue':
            wet_column = 'Bs_B_Wet_1um_Neph3W_2'
            dry_column = 'Bs_B_Dry_1um_Neph3W_1'
        else:
            txt = '%s is not an option. Choose between ["all", "green", "red", "blue"]'
            raise ValueError(txt)


        scatt_coeff_wet = noaaaos.scatt_coeff.data[wet_column].copy()
        scatt_coeff_dry = noaaaos.scatt_coeff.data[dry_column].copy()

        scatt_coeff_wet.values[_np.isnan(rh_wet.values)] = _np.nan
        scatt_coeff_dry.values[_np.isnan(scatt_coeff_wet.values)] = _np.nan
        f_rh = scatt_coeff_wet/scatt_coeff_dry

        f_rh_int = f_rh.interpolate()
        f_rh_mean = _pd.rolling_mean(f_rh_int, 40, center = True)
        df[col] = f_rh_mean
    ts = _timeseries.TimeSeries(df)
    ts._y_label = '$f(RH = %i \pm %i \%%)$'%(RH_center, RH_tolerance)
    return ts




class ArmDatasetSub(_ArmDataset):
    def __init__(self,*args, **kwargs):
        super(ArmDatasetSub,self).__init__(*args, **kwargs)
        self.__f_of_RH = None
        self.__kappa = None
        self.__growthfactor = None

        self.__sup_fofRH_RH_center = None
        self.__sup_fofRH_RH_tolerance = None
        self.__sup_fofRH_which = None
        self.__sup_kappa_sizedist = None
        self.__sup_kappa_wavelength = None

    @property
    def sup_kappa_wavelength(self):
        return self.__sup_kappa_wavelength

    @sup_kappa_wavelength.setter
    def sup_kappa_wavelength(self, value):
        self.__kappa = None
        self.__growthfactor = None
        self.__sup_kappa_wavelength = value

    @property
    def sup_kappa_sizedist(self):
        return self.__sup_kappa_sizedist

    @sup_kappa_sizedist.setter
    def sup_kappa_sizedist(self, value):
        self.__kappa = None
        self.__growthfactor = None
        self.__sup_kappa_sizedist = value

    @property
    def sup_fofRH_which(self):
        return self.__sup_fofRH_which

    @sup_fofRH_which.setter
    def sup_fofRH_which(self, value):
        self.__f_of_RH = None
        self.__sup_fofRH_which = value

    @property
    def sup_fofRH_RH_center(self):
        return self.__sup_fofRH_RH_center

    @sup_fofRH_RH_center.setter
    def sup_fofRH_RH_center(self, value):
        self.__f_of_RH = None
        self.__sup_fofRH_RH_center = value

    @property
    def sup_fofRH_RH_tolerance(self):
        return self.__sup_fofRH_RH_tolerance

    @sup_fofRH_RH_tolerance.setter
    def sup_fofRH_RH_tolerance(self, value):
        self.__f_of_RH = None
        self.__sup_fofRH_RH_tolerance = value

    def _parse_netCDF(self):

        abs_coeff = ['Ba_G_Dry_10um_PSAP1W_1',
                    'Ba_G_Dry_1um_PSAP1W_1',
                    'Ba_B_Dry_10um_PSAP3W_1',
                    'Ba_G_Dry_10um_PSAP3W_1',
                    'Ba_R_Dry_10um_PSAP3W_1',
                    'Ba_B_Dry_1um_PSAP3W_1',
                    'Ba_G_Dry_1um_PSAP3W_1',
                    'Ba_R_Dry_1um_PSAP3W_1',
                    ]

        scat_coeff =   ['Bs_B_Dry_10um_Neph3W_1',
                            'Bs_G_Dry_10um_Neph3W_1',
                            'Bs_R_Dry_10um_Neph3W_1',
                            'Bs_B_Wet_10um_Neph3W_2',
                            'Bs_G_Wet_10um_Neph3W_2',
                            'Bs_R_Wet_10um_Neph3W_2',
                            'Bs_B_Dry_1um_Neph3W_1',
                            'Bs_G_Dry_1um_Neph3W_1',
                            'Bs_R_Dry_1um_Neph3W_1',
                            'Bs_B_Wet_1um_Neph3W_2',
                            'Bs_G_Wet_1um_Neph3W_2',
                            'Bs_R_Wet_1um_Neph3W_2',
                            ]


        bscat_coeff_vars = ['Bbs_B_Dry_10um_Neph3W_1',
                            'Bbs_G_Dry_10um_Neph3W_1',
                            'Bbs_R_Dry_10um_Neph3W_1',
                            'Bbs_B_Wet_10um_Neph3W_2',
                            'Bbs_G_Wet_10um_Neph3W_2',
                            'Bbs_R_Wet_10um_Neph3W_2',
                            'Bbs_B_Dry_1um_Neph3W_1',
                            'Bbs_G_Dry_1um_Neph3W_1',
                            'Bbs_R_Dry_1um_Neph3W_1',
                            'Bbs_B_Wet_1um_Neph3W_2',
                            'Bbs_G_Wet_1um_Neph3W_2',
                            'Bbs_R_Wet_1um_Neph3W_2',
                            ]

        RH_neph = ['RH_NephVol_Dry',
              'RH_NephVol_Wet']

        def var2ts(self, var_list, column_name):
            """extracts the list of variables from the file_obj and puts them all in one data frame"""
            df = _pd.DataFrame(index = self.time_stamps)
            for var in var_list:
                data = self._read_variable(var)
                # variable = file_obj.variables[var]
                # data = variable[:]
                # fill_value = variable.missing_data
                # data = np.ma.masked_where(data == fill_value, data)
                df[var] = _pd.Series(data, index = self.time_stamps)
            df.columns.name = column_name
            out = _timeseries.TimeSeries(df)
            return out

        # out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
        # out['abs_coeff'] = var2ts(self, abs_coeff, 'abs_coeff_1/Mm')
        # out['scatt_coeff'] = var2ts(self, scat_coeff, 'scatt_coeff_1/Mm')
        # out['back_scatt'] = var2ts(self, bscat_coeff_vars, 'back_scatt_1/Mm')
        # out['RH'] = var2ts(self, RH, 'RH')
        # out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
        self.abs_coeff = var2ts(self, abs_coeff, 'abs_coeff_1/Mm')
        self.scatt_coeff = var2ts(self, scat_coeff, 'scatt_coeff_1/Mm')
        self.back_scatt = var2ts(self, bscat_coeff_vars, 'back_scatt_1/Mm')
        self.RH_nephelometer = var2ts(self, RH_neph, 'RH')
        # _pdb.set_trace()
        # df = pd.DataFrame(self.read_variable('RH_interDMA'), index = self.time_stamps, columns=size_bins)
        # df.columns.name = 'size_bin_center_nm'
        # self.RH_interDMA = timeseries.TimeSeries(df)
        # self.RH_ambient = self.read_variable(timeseries.TimeSeries('RH_Ambient'))
        # self.temperature_ambient = self.read_variable('T_Ambient')


    def plot_all(self):
        self.abs_coeff.plot()
        self.back_scatt.plot()
        self.scatt_coeff.plot()
        self.RH_nephelometer.plot()


    @property
    @decorators.change_doc(calculate_f_RH)
    def f_of_RH(self):
        """
        Parameter names are changed to self.sup_fofRH_RH_center, self.sup_fofRH_RH_tolerance, and self.sup_fofRH_which
        """
        if not self.__f_of_RH:
            if not self.sup_fofRH_RH_center or not self.sup_fofRH_RH_tolerance or not self.sup_fofRH_which:
                txt = "Make sure you define the following attributes first: \nself.sup_fofRH_RH_center, self.sup_fofRH_RH_tolerance, self.sup_fofRH_which"
                raise ValueError(txt)

            self.__f_of_RH = calculate_f_RH(self,self.sup_fofRH_RH_center, self.sup_fofRH_RH_tolerance, self.sup_fofRH_which)

        return self.__f_of_RH

    @property
    @decorators.change_doc(hygrow.kappa_from_fofrh_and_sizedist)
    def kappa(self):
        if not self.__kappa:
            if not self.sup_kappa_sizedist or not self.sup_kappa_wavelength:
                txt = "Make sure you define the following attributes first: \nself.sup_kappa_sizedist and self.sup_kappa_wavelength"
                raise ValueError(txt)

            self.__kappa, self.__growthfactor = hygrow.kappa_from_fofrh_and_sizedist(self.f_of_RH, self.sup_kappa_sizedist,
                                                                                     self.sup_kappa_wavelength, self.sup_fofRH_RH_center)
        return self.__kappa

    @property
    @decorators.change_doc(hygrow.kappa_from_fofrh_and_sizedist)
    def growth_factor(self):
        if self.__growthfactor:
            self.kappa
        return self.__growthfactor



def _concat_rules(arm_data_objs):
    out = ArmDatasetSub(False)
    out.abs_coeff = _timeseries.TimeSeries(_pd.concat([i.abs_coeff.data for i in arm_data_objs]))
    out.back_scatt = _timeseries.TimeSeries(_pd.concat([i.back_scatt.data for i in arm_data_objs]))
    out.scatt_coeff = _timeseries.TimeSeries(_pd.concat([i.scatt_coeff.data for i in arm_data_objs]))
    out.RH_nephelometer = _timeseries.TimeSeries(_pd.concat([i.RH_nephelometer.data for i in arm_data_objs]))
    # out.RH_ambient = timeseries.TimeSeries(pd.concat([i.RH_ambient.data for i in arm_data_objs]))
    # out.temperature_ambient = timeseries.TimeSeries(pd.concat([i.temperature_ambient.data for i in arm_data_objs]))
    out.time_stamps = out.abs_coeff.data.index
    # out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
    # out['abs_coeff'] = timeseries.TimeSeries(pd.concat([i['abs_coeff'].data for i in files]))
    # out['scatt_coeff'] = timeseries.TimeSeries(pd.concat([i['scatt_coeff'].data for i in files]))
    # out['back_scatt'] = timeseries.TimeSeries(pd.concat([i['back_scatt'].data for i in files]))
    # out['RH'] = timeseries.TimeSeries(pd.concat([i['RH'].data for i in files]))
    return out


