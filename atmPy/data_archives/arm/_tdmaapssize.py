from atmPy.aerosols.size_distribution import sizedistribution
from atmPy.aerosols.size_distribution import diameter_binning
import pandas as pd
from atmPy.data_archives.arm._netCDF import ArmDataset


class ArmDatasetSub(ArmDataset):
    def __init__(self,*args, **kwargs):
        super(ArmDatasetSub,self).__init__(*args, **kwargs)

    def _parse_netCDF(self):
        super(ArmDatasetSub,self)._parse_netCDF()


# def _parse_netCDF(file_obj):


        # sd = file_obj.variables['number_concentration_DMA_APS']
        df = pd.DataFrame(self.read_variable('number_concentration_DMA_APS'),
                          index = self.time_stamps)

        d = self.read_variable('diameter')
        bins, colnames = diameter_binning.bincenters2binsANDnames(d[:]*1000)

        self.size_distribution = sizedistribution.SizeDist_TS(df,bins,'dNdlogDp')

    def plot_all(self):
        self.size_distribution.plot()

def _concat_rules(files):
    out = ArmDatasetSub(False)
    data = pd.concat([i.size_distribution.data for i in files])
    out.size_distribution = sizedistribution.SizeDist_TS(data,files[0].size_distribution.bins,'dNdlogDp')
    return out
