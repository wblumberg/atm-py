from hagpack.projects.arm import _tools
import pandas as pd
from atmPy import timeseries


def _parse_netCDF(file_obj):
    index = _tools._get_time(file_obj)
    mass_concentrations = pd.DataFrame(index = index)
    mass_conc_keys = ['total_organics','ammonium','sulfate','nitrate','chloride']

    for k in mass_conc_keys:
        var = file_obj.variables[k]
        data = var[:]
        mass_concentrations[k] = pd.Series(var, index = index)

    mass_concentrations.columns.name = 'Mass conc. ug/m^3'
    mass_concentrations.index.name = 'Time'

    # for v in file_obj.variables.keys():
    #     var = file_obj.variables[v]
    #     print(v)
    #     print(var.long_name)
    #     print(var.shape)
    #     print('--------')

    org_mx = file_obj.variables['org_mx'][:]
    org_mx = pd.DataFrame(org_mx, index = index)
    org_mx.index.name = 'Time'
    org_mx.columns = file_obj.variables['amus'][:]
    org_mx.columns.name = 'amus (m/z)'

    out = _tools.ArmDict(plottable = ['mass_concentrations', 'Organic mass spectral matrix'])
    out['mass_concentrations'] = timeseries.TimeSeries(mass_concentrations)
    out['Organic mass spectral matrix'] = timeseries.TimeSeries_2D(org_mx)
    return out

def _concat_rules(files):
    out = _tools.ArmDict(plottable = ['mass_concentrations', 'Organic mass spectral matrix'])
    out['mass_concentrations'] = timeseries.TimeSeries(pd.concat([i['mass_concentrations'].data for i in files]))
    out['Organic mass spectral matrix'] = timeseries.TimeSeries_2D(pd.concat([i['Organic mass spectral matrix'].data for i in files]))
    return out