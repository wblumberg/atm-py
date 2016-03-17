from netCDF4 import Dataset
import numpy as np
import pandas as pd
from atmPy.general import timeseries as _timeseries
from atmPy.tools import array_tools as _arry_tools

class ArmDataset(object):
    def __init__(self, fname, data_quality = 'good', data_quality_flag_max = None):
        if fname:
            self.netCDF = Dataset(fname)
            self.data_quality_flag_max = data_quality_flag_max
            self.data_quality = data_quality

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

    def _read_variable(self, variable, reverse_qc_flag = False):
        """Reads the particular variable and replaces all masked data with NaN.
        Note, if quality flag is given only values larger than the quality_control
        variable are replaced with NaN.

        Parameters
        ----------
        variable: str
            Variable name as devined in netCDF file
        reverse_qc_flag: bool or int
            Set to the number of bits, when reversion is desired
            If the indeterminate (patchy) bits are on the wrong end of the qc bit
            string it might make sense to reverte the bit string.

        Returns
        -------
        ndarray

        Examples
        --------
        self.temp = self.read_variable(ti"""
        var = self.netCDF.variables[variable]
        data = var[:]


        variable_qc = "qc_" + variable
        if variable_qc in self.netCDF.variables.keys():
            # print('has qc')
            var_qc = self.netCDF.variables["qc_" + variable]
            data_qc = var_qc[:]
            if reverse_qc_flag:
                if type(reverse_qc_flag) != int:
                    raise TypeError('reverse_qc_flag should either be False or of type integer giving the number of bits')
                data_qc = _arry_tools.reverse_binary(data_qc, reverse_qc_flag)
            data_qc[data_qc <= self.data_quality_flag_max] = 0
            data_qc[data_qc > self.data_quality_flag_max] = 1

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

    def _read_variable2timeseries(self, variable, column_name = False):
        """
        Reads the specified variables and puts them into a timeseries.

        Parameters
        ----------
        variable: string or list of strings
            variable names
        column_name: bool or string
            this is a chance to give unites. This will also be the y-label if data
            is plotted

        Returns
        -------
        pandas.DataFrame

        """


        if type(variable).__name__ == 'str':
            variable = [variable]

        df = pd.DataFrame(index = self.time_stamps)
        for var in variable:
            data = self._read_variable(var)
            df[var] = pd.Series(data, index = self.time_stamps)
        if column_name:
            df.columns.name = column_name
        out = _timeseries.TimeSeries(df)
        if column_name:
            out._y_label = column_name
        return out


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
