import pandas as pd
from atmPy.general import timeseries
from atmPy.aerosols.instruments.AMS import AMS
from atmPy.data_archives.arm._netCDF import ArmDataset

class ArmDatasetSub(ArmDataset):
    def __init__(self,*args, **kwargs):
        super(ArmDatasetSub,self).__init__(*args, **kwargs)

    def _parse_netCDF(self):
        super(ArmDatasetSub,self)._parse_netCDF()
        mass_concentrations = pd.DataFrame(index = self.time_stamps)
        mass_conc_keys = ['total_organics','ammonium','sulfate','nitrate','chloride']

        for k in mass_conc_keys:
            mass_concentrations[k] = pd.Series(self.read_variable(k), index = self.time_stamps)

        mass_concentrations.columns.name = 'Mass conc. ug/m^3'
        mass_concentrations.index.name = 'Time'

        # for v in file_obj.variables.keys():
        #     var = file_obj.variables[v]
        #     print(v)
        #     print(var.long_name)
        #     print(var.shape)
        #     print('--------')

        org_mx = self.read_variable('org_mx')
        org_mx = pd.DataFrame(org_mx, index = self.time_stamps)
        org_mx.columns = self.read_variable('amus')
        org_mx.columns.name = 'amus (m/z)'

        # out = _tools.ArmDict(plottable = ['mass_concentrations', 'Organic mass spectral matrix'])
        # out['mass_concentrations'] = timeseries.TimeSeries(mass_concentrations)
        # out['Organic mass spectral matrix'] = timeseries.TimeSeries_2D(org_mx)

        self.mass_concentrations = AMS.AMS_Timeseries(mass_concentrations)
        self.organic_mass_spectral_matrix = timeseries.TimeSeries_2D(org_mx)
        return

    @property
    def mass_concentration_corr(self):
        if '__mass_concentration_corr' not in dir(self):
            self.__mass_concentration_corr = self.mass_concentrations.calculate_electrolyte_mass_concentrations()
        return self.__mass_concentration_corr

    def plot_all(self):
        self.mass_concentrations.plot()
        self.organic_mass_spectral_matrix.plot()

def _concat_rules(arm_data_objs):
    # out = arm_data_obj
    out = ArmDatasetSub(False)
    out.mass_concentrations = AMS.AMS_Timeseries(
        pd.concat([i.mass_concentrations.data for i in arm_data_objs]))
    out.organic_mass_spectral_matrix = timeseries.TimeSeries_2D(pd.concat([i.organic_mass_spectral_matrix.data for i in arm_data_objs]))

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