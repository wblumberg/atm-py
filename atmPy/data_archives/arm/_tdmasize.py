import pandas as pd
from atmPy.data_archives.arm import _tools
from atmPy.aerosols.size_distribution import diameter_binning
from atmPy.aerosols.size_distribution import sizedistribution

def _parse_netCDF(file_obj):


    index = _tools._get_time(file_obj)

    sd = file_obj.variables['number_concentration']
    df = pd.DataFrame(sd[:])
    df.index = index

    d = file_obj.variables['diameter']
    bins, colnames = diameter_binning.bincenters2binsANDnames(d[:]*1000)

    dist = sizedistribution.SizeDist_TS(df,bins,'dNdlogDp')
    dist = dist.convert2dVdlogDp()

    out = _tools.ArmDict(plottable = ['size_distribution'])
    out['size_distribution'] = dist

    return out

def _concat_rules(files):
    dist = files[0]['size_distribution']
    dist.data = pd.concat([i['size_distribution'].data for i in files])
    files[0]['size_distribution'] = dist
    return files[0]