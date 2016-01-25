import pandas as pd
from hagpack.projects.arm import _tools
import numpy as np
from atmPy import timeseries
from atmPy.tools import math_functions
from scipy.optimize import curve_fit
from atmPy.aerosols import hygroscopic_growth as hg


class Tdmahyg(_tools.ArmDict):
    def __init__(self,*args,**kwargs):
        super(Tdmahyg,self).__init__(*args,**kwargs)

    def calc_mean_growth_factor(self):
        """Calculates the mean growthfactor of the particular size bin."""
        def mean_linewise(gf_dist):
            growthfactors = self['hyg_distributions'].data.minor_axis.values
            meanl = ((gf_dist[~ gf_dist.mask] * np.log10(growthfactors[~ gf_dist.mask])).sum()/gf_dist[~gf_dist.mask].sum())
            stdl = np.sqrt((gf_dist[~ gf_dist.mask] * (np.log10(growthfactors[~ gf_dist.mask]) - meanl)**2).sum()/gf_dist[~gf_dist.mask].sum())
            return np.array([10**meanl,stdl])
        data = self['hyg_distributions'].data
        allmeans = timeseries.TimeSeries_3D(pd.Panel(items=data.items, major_axis=data.major_axis, minor_axis= ['mean', 'std_log']))
        for i,time in enumerate(data.values):
            for e,size in enumerate(time):
                allmeans.data.iloc[i,e] = mean_linewise(size)
        self['allmeans'] = allmeans
        return allmeans

    def calc_kappa_values(self):
        if not 'allmeans' in self.keys():
            allmeans = self.calc_mean_growth_factor()
        else:
            allmeans = self['allmeans']

        RH = self['RH_interDMA']
        kappa_values = hg.kappa_simple(allmeans.data.values[:,:,0],RH, inverse = True)

        kappa_values = pd.DataFrame(kappa_values,columns=allmeans.data.major_axis, index = allmeans.data.items)
        out = timeseries.TimeSeries_2D(kappa_values)
        self['kappa_values'] = out
        self.plottable.append('kappa_values')
        return out

    def _fit_growth_factor(self, data):
        """Not recommended, probably not working"""
        def fit_linewise(gf_dist):
            fkt = math_functions.gauss
            amp = gf_dist.max()
            growthfactors = self['hyg_distributions'].data.minor_axis.values
            pos = growthfactors[gf_dist.argmax()]
            sigma = 0.4
            cof,varm = curve_fit(fkt,growthfactors[~gf_dist.mask], gf_dist[~gf_dist.mask], p0=[amp,pos,sigma])
            return np.array(cof)

        shape = list(data.shape)
        shape[-1] = 3
        fitres = np.zeros(shape)
        for i,time in enumerate(data):
            for e,size in enumerate(time):
                fitres[i,e] = fit_linewise(size)
        return fitres

def _parse_netCDF(file_obj):
    "returns a dictionary, with panels in it"
    index = _tools._get_time(file_obj)
    data = file_obj.variables['hyg_distributions'][:]
    growthfactors = file_obj.variables['growthfactors'][:]
    size_bins = file_obj.variables['size_bins'][:]* 1000
    RH_interDMA = pd.DataFrame(file_obj.variables['RH_interDMA'][:], index = index, columns=size_bins)
    RH_interDMA.columns.name = 'size_bin_center_nm'

    data = pd.Panel(data, items= index, major_axis = size_bins, minor_axis = growthfactors)
    data.items.name = 'Time'
    data.major_axis.name = 'size_bin_center_nm'
    data.minor_axis.name = 'growthfactors'

    out = Tdmahyg(plottable = ['hyg_distributions'], plot_kwargs =  dict(xaxis=0, yaxis = 2, sub_set=5, kwargs = dict(vmin = 0)))

#     data = timeseries.TimeSeries_3D(data)
    data = timeseries.TimeSeries_3D(data)
#     data.RH_interDMA = RH_interDMA
    out['hyg_distributions'] = data
    out['RH_interDMA'] = RH_interDMA
#     out['growthfactors'] = growthfactors
#     out['size_bins'] = size_bins
    return out

def _concat_rules(files):
    out = Tdmahyg(plottable = ['hyg_distributions'], plot_kwargs =  dict(xaxis=0, yaxis = 2, sub_set=5, kwargs = dict(vmin = 0)))
#     data = timeseries.TimeSeries_3D(pd.concat([i['hyg_distributions'].data for i in files]))
    data = pd.concat([i['hyg_distributions'].data for i in files])
    data.iloc[:] = np.ma.masked_array(data.values, mask = data.values == -9999.0, fill_value = -9999.0)
    ts = timeseries.TimeSeries_3D(data)
#     data.RH_interDMA = RH_interDMA
    out['hyg_distributions'] = ts
    out['RH_interDMA'] = pd.concat([i['RH_interDMA'] for i in files])
    return out
