from netCDF4 import Dataset
import numpy as np
import pandas as pd

class ArmDataset(object):
    def __init__(self, fname, quality_control = 0):
        if fname:
            self.netCDF = Dataset(fname)
            self.quality_control = quality_control
            self._parse_netCDF()

    @property
    def time_stamps(self):
        if '__time_stamps' in dir(self):
            return self.__time_stamps
        else:
            bt = self.netCDF.variables['base_time']
            toff = self.netCDF.variables['time_offset']
            self.__time_stamps = pd.to_datetime(0) + pd.to_timedelta(bt[:].flatten()[0], unit = 's') + pd.to_timedelta(toff[:], unit = 's')
            self.__time_stamps.name = 'Time'
        return self.__time_stamps

    @time_stamps.setter
    def time_stamps(self,timesamps):
        self.__time_stamps = timesamps

    def read_variable(self, variable):
        var = self.netCDF.variables[variable]
        data = var[:]


        variable_qc = "qc_" + variable
        if variable_qc in self.netCDF.variables.keys():
            # print('has qc')
            var_qc = self.netCDF.variables["qc_" + variable]
            data_qc = var_qc[:]
            data_qc[data_qc <= self.quality_control] = 0
            data_qc[data_qc > self.quality_control] = 1

            if data.shape != data_qc.shape:
                dt = np.zeros(data.shape)
                dt[data_qc == 1, : ] = 1
                data_qc = dt
            data = np.ma.array(data, mask = data_qc, fill_value= -9999)

        elif 'missing_data' in var.ncattrs():
            # print('has missing data')
            fill_value = var.missing_data
            data = np.ma.masked_where(data == fill_value, data)
        # else:
            # print('no quality flag found')

        if type(data).__name__ == 'MaskedArray':
            data.data[data.mask] = np.nan
            data = data.data
        return data

    def get_variable_info(self):
        for v in self.netCDF.variables.keys():
            var = self.netCDF.variables[v]
            print(v)
            print(var.long_name)
            print(var.shape)
            print('--------')

    def close(self):
        self.netCDF.close()

    def _parse_netCDF(self):
        return
