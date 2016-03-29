from atmPy.general import timeseries as _timeseries
from atmPy.data_archives.arm import _netCDF
import pandas as _pd
import numpy as _np

class ArmDatasetSub(_netCDF.ArmDataset):
    def __init__(self,*args, **kwargs):
        self._data_period = 3600
        super(ArmDatasetSub,self).__init__(*args, **kwargs)

        self._concatable = ['f_RH_scatt_funcs_2p', 'f_RH_scatt_funcs_3p']


        ####
        # for properties
        self.__f_RH_scatt_2p = None
        self.__f_RH_scatt_3p  = None
        self.__sup_RH = None


    def _data_quality_control(self):
        #######
        ## Define what is good, patchy or bad data
        ## delete if not quality flags exist
        ## settings are ignored if no quality flags are profided (e.g. noaaaos)

        if self.data_quality_flag_max == None:
            if self.data_quality == 'good':
                self.data_quality_flag_max = 0
            elif self.data_quality == 'patchy':
                self.data_quality_flag_max = 7
            elif self.data_quality == 'bad':
                self.data_quality_flag_max = 100000
            else:
                txt = '%s is not an excepted values for data_quality ("good", "patchy", "bad")'%(self.data_quality)
                raise ValueError(txt)

    def _parse_netCDF(self):
        super(ArmDatasetSub,self)._parse_netCDF()
        # self.rh = self._read_variable2timeseries(['rh_60m', 'rh_60m'], column_name='Relative Humidity (%)')


        # for the 2 parameter function
        def ab_2_f_RH_func(ab):
            ab = ab.copy()
            a, b = ab
            f_RH = lambda RH: a * (1 - (RH / 100.)) ** (-b)  # 'bsp(RH%)/Bsp(~40%) = a*[1-(RH%/100)]^(-b)'
            return f_RH

        varies = ['fRH_Bs_R_10um_2p',
                  'fRH_Bs_G_10um_2p',
                  'fRH_Bs_B_10um_2p',
                  'fRH_Bs_R_1um_2p',
                  'fRH_Bs_G_1um_2p',
                  'fRH_Bs_B_1um_2p']

        df = _pd.DataFrame(index=self.time_stamps)
        for key in varies:
            data = self._read_variable(key, reverse_qc_flag=8)
            dft = _pd.DataFrame(data, index=self.time_stamps)
            df[key] = dft.apply(ab_2_f_RH_func, axis=1)
        self.f_RH_scatt_funcs_2p = _timeseries.TimeSeries(df)
        self.f_RH_scatt_funcs_2p._data_period = self._data_period


        #for the 3 parameter function
        def abc_2_f_RH_func(abc):
            abc = abc.copy()
            a, b, c = abc
            f_RH = lambda RH: a * (1 + (b * (RH / 100.)**c))
            return f_RH

        varies = ['fRH_Bs_R_10um_3p',
                  'fRH_Bs_G_10um_3p',
                  'fRH_Bs_B_10um_3p',
                  'fRH_Bs_R_1um_3p',
                  'fRH_Bs_G_1um_3p',
                  'fRH_Bs_B_1um_3p']

        df = _pd.DataFrame(index=self.time_stamps)
        for key in varies:
            data = self._read_variable(key, reverse_qc_flag=8)
            dft = _pd.DataFrame(data, index=self.time_stamps)
            df[key] = dft.apply(abc_2_f_RH_func, axis=1)
        self.f_RH_scatt_funcs_3p = _timeseries.TimeSeries(df)
        self.f_RH_scatt_funcs_3p._data_period = self._data_period

    def plot_all(self):
        self.rh.plot()

    @property
    def f_RH_scatt_3p(self):
        """do something more with the data"""
        if not self.__f_RH_scatt_3p:
            if not self.sup_RH:
                raise ValueError('please set the relative humidity in sup_RH')

            def applyfunk(value):
                if type(value).__name__ == 'function':
                    return value(self.sup_RH)
                else:
                    return _np.nan

            # data = self.f_RH_scatt_funcs.data.applymap(lambda x: x(self.sup_RH))
            data = self.f_RH_scatt_funcs_3p.data.applymap(applyfunk)
            # data = _pd.DataFrame(data, columns=['f_%i'%(self.sup_RH)])
            self.__f_RH_scatt_3p = _timeseries.TimeSeries(data)
            self.__f_RH_scatt_3p._data_period = self.f_RH_scatt_funcs_3p._data_period
        return self.__f_RH_scatt_3p

    @property
    def f_RH_scatt_2p(self):
        """do something more with the data"""
        if not self.__f_RH_scatt_2p:
            if not self.sup_RH:
                raise ValueError('please set the relative humidity in sup_RH')

            def applyfunk(value):
                if type(value).__name__ == 'function':
                    return value(self.sup_RH)
                else:
                    return _np.nan

            # data = self.f_RH_scatt_funcs.data.applymap(lambda x: x(self.sup_RH))
            data = self.f_RH_scatt_funcs_2p.data.applymap(applyfunk)
            # data = _pd.DataFrame(data, columns=['f_%i'%(self.sup_RH)])
            self.__f_RH_scatt_2p = _timeseries.TimeSeries(data)
            self.__f_RH_scatt_2p._data_period = self.f_RH_scatt_funcs_2p._data_period
        return self.__f_RH_scatt_2p

    @property
    def sup_RH(self):
        return self.__sup_RH

    @sup_RH.setter
    def sup_RH(self, value):
        self.__sup_RH = value
        self.__f_RH_scatt_3p = None
        self.__f_RH_scatt_2p = None



def _concat_rules(arm_data_objs):
    """nothing here"""
    # out = arm_data_obj
    out = ArmDatasetSub(False)
    out._concat(arm_data_objs)
    return out

# def _concat_rules(arm_data_objs):
#     # create class
#     out = ArmDatasetSub(False)
#
#     # populate class with concatinated data
#     out.rh = _timeseries.concat([i.rh for i in arm_data_objs])
#     out.rh._data_periode = out._data_periode
#
#     # use time stamps from one of the variables
#     out.time_stamps = out.rh.data.index
#     return out
