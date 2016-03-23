import pandas as _pd
from atmPy.general import timeseries as _timeseries
from atmPy.aerosols.instruments.AMS import AMS as _AMS
from atmPy.data_archives.arm._netCDF import ArmDataset as _ArmDataset
from atmPy.tools import decorators as _decorators

def _concat_rules(arm_data_objs):
    """nothing here"""
    # out = arm_data_obj
    out = ArmDatasetSub(False)
    out.mass_concentrations = _AMS.AMS_Timeseries_lev01(
        _pd.concat([i.mass_concentrations.data for i in arm_data_objs]))
    # out.mass_concentrations._data_periode = out._data_periode
    out.organic_mass_spectral_matrix = _timeseries.TimeSeries_2D(_pd.concat([i.organic_mass_spectral_matrix.data for i in arm_data_objs]))
    # out.organic_mass_spectral_matrix._data_periode = out._data_periode

    # out = _tools.ArmDict(plottable = ['mass_concentrations', 'Organic mass spectral matrix'])
    # out['mass_concentrations'] = timeseries.TimeSeries(pd.concat([i['mass_concentrations'].data for i in files]))
    # out['Organic mass spectral matrix'] = timeseries.TimeSeries_2D(pd.concat([i['Organic mass spectral matrix'].data for i in files]))
    return out


        # var = file_obj.variables[k]
        # data = var[:]
        # var_qc = file_obj.variables["qc_" + k]
        # data_qc = var_qc[:]
        # data = np.ma.array(data, mask = data_qc, fill_value= -9999)
        #
        # if any(data_qc > 2):
        #     txt = "I was not aware of a quality control level %s. Maximum is 2."%quality_control
        #     raise ValueError(txt)
        # data_qc[data_qc <= quality_control] = 0
        # data_qc[data_qc > quality_control] = 1
        # mass_concentrations[k] = pd.Series(data, index = index)

class ArmDatasetSub(_ArmDataset):
    def __init__(self,*args, **kwargs):
        super(ArmDatasetSub,self).__init__(*args, **kwargs)

        self._data_periode = 2048.

        self.__kappa = None
        self.__mass_concentration_corr = None
        self.__refractive_index = None
        self.__density = None
        self.__mass_concentration_corr_relative = None

    def _data_quality_control(self):
        ## Define what is good, patchy or bad data
        if self.data_quality_flag_max == None:
            if self.data_quality == 'good':
                self.data_quality_flag_max = 0
            elif self.data_quality == 'patchy':
                self.data_quality_flag_max = 1
            elif self.data_quality == 'bad':
                self.data_quality_flag_max = 100000
            else:
                txt = '%s is not an excepted values for data_quality ("good", "patchy", "bad")'%(self.data_quality)
                raise ValueError(txt)

    def _parse_netCDF(self):
        super(ArmDatasetSub,self)._parse_netCDF()
        mass_concentrations = _pd.DataFrame(index = self.time_stamps)
        mass_conc_keys = ['total_organics','ammonium','sulfate','nitrate','chloride']

        for k in mass_conc_keys:
            mass_concentrations[k] = _pd.Series(self._read_variable(k, reverse_qc_flag = 4), index = self.time_stamps)

        mass_concentrations.columns.name = 'Mass conc. ug/m^3'
        mass_concentrations.index.name = 'Time'

        org_mx = self._read_variable('org_mx')
        org_mx = _pd.DataFrame(org_mx, index = self.time_stamps)
        org_mx.columns = self._read_variable('amus')
        org_mx.columns.name = 'amus (m/z)'

        self.mass_concentrations = _AMS.AMS_Timeseries_lev01(mass_concentrations)
        self.mass_concentrations.data['total'] = self.mass_concentrations.data.sum(axis = 1)
        self.mass_concentrations.data.rename(columns= {'total_organics': 'organic_aerosol'}, inplace = True)
        self.mass_concentrations._data_periode = self._data_periode

        self.organic_mass_spectral_matrix = _timeseries.TimeSeries_2D(org_mx)
        self.organic_mass_spectral_matrix._data_periode = self._data_periode
        return


    @property
    @_decorators.change_doc(_AMS.AMS_Timeseries_lev02.calculate_density, add_warning=False)
    def density(self):
        if self.__density is None:
            self.__density = self.mass_concentration_corr.calculate_density()
            self.__density._data_periode = self._data_periode
        return self.__density


    @property
    @_decorators.change_doc(_AMS.AMS_Timeseries_lev02.calculate_kappa, add_warning=False)
    def kappa(self):
        if self.__kappa is None:
            self.__kappa = self.mass_concentration_corr.calculate_kappa()
            self.__kappa._data_periode = self._data_periode
        return self.__kappa

    @kappa.setter
    def kappa(self,value):
        self.__kappa = value

    @property
    def mass_concentration_corr(self):
        if self.__mass_concentration_corr is None:
            self.__mass_concentration_corr = self.mass_concentrations.calculate_electrolyte_mass_concentrations()
            self.__mass_concentration_corr.data['total'] = self.__mass_concentration_corr.data.sum(axis = 1)
            self.__mass_concentration_corr._data_periode = self._data_periode
        return self.__mass_concentration_corr

    @property
    def mass_concentration_corr_relative(self):
        if self.__mass_concentration_corr_relative is None:
            mccr = self.mass_concentration_corr.copy()
            total = mccr.data.loc[:,'total']
            mccr.data.drop(['total'], axis=1, inplace=True)
            mccr.data = mccr.data.divide(total, axis = 0)
            self.__mass_concentration_corr_relative = mccr
        return self.__mass_concentration_corr_relative

    @property
    @_decorators.change_doc(_AMS.AMS_Timeseries_lev02.calculate_refractive_index, add_warning=False)
    def refractive_index(self):
        if self.__refractive_index is None:
            self.__refractive_index = self.mass_concentration_corr.calculate_refractive_index()
            self.__refractive_index._data_periode = self._data_periode
        return self.__refractive_index


    def plot_all(self):
        self.mass_concentrations.plot()
        self.organic_mass_spectral_matrix.plot()

