from atmPy.general import timeseries as _timeseries
from atmPy.data_archives.arm import _netCDF


class ArmDatasetSub(_netCDF.ArmDataset):
    def __init__(self,*args, **kwargs):
        super(ArmDatasetSub,self).__init__(*args, **kwargs)

        ####
        # for properties
        self.__mean_growth_factor  = None

    def _parse_netCDF(self):
        super(ArmDatasetSub,self)._parse_netCDF()
        self.rh = self._read_variable2timeseries(['rh_60m', 'rh_60m'], column_name='Relative Humidity (%)')

    def plot_all(self):
        self.rh.plot()

    @property
    def mean_growth_factor(self):
        """do something more with the data"""
        if '__mean_growth_factor' not in dir(self):
            self.__mean_growth_factor = 'bla'
        return self.__mean_growth_factor


def _concat_rules(arm_data_objs):
    # create class
    out = ArmDatasetSub(False)

    # populate class with concatinated data
    out.rh = _timeseries.concat([i.rh for i in arm_data_objs])

    # use time stamps from one of the variables
    out.time_stamps = out.rh.data.index
    return out
