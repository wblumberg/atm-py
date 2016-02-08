import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from atmPy.aerosols.physics import hygroscopic_growth as hg
from atmPy.data_archives.arm import _tools
from atmPy.general import timeseries
from atmPy.tools import math_functions
from atmPy.data_archives.arm._netCDF import ArmDataset



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

class ArmDatasetSub(ArmDataset):
    def __init__(self,*args, **kwargs):
        super(ArmDatasetSub,self).__init__(*args, **kwargs)


    def _parse_netCDF(self):
        "returns a dictionary, with panels in it"
        super(ArmDatasetSub,self)._parse_netCDF()


        size_bins = self.read_variable('size_bins') * 1000
        df = pd.DataFrame(self.read_variable('RH_interDMA'), index = self.time_stamps, columns=size_bins)
        df.columns.name = 'size_bin_center_nm'
        self.RH_interDMA = timeseries.TimeSeries(df)

        data = self.read_variable('hyg_distributions')
        growthfactors = self.read_variable('growthfactors')
        data = pd.Panel(data, items= self.time_stamps, major_axis = size_bins, minor_axis = growthfactors)
        data.major_axis.name = 'size_bin_center_nm'
        data.minor_axis.name = 'growthfactors'
        self.hyg_distributions = timeseries.TimeSeries_3D(data)

    def plot_all(self):
        self.hyg_distributions.plot(yaxis=2, sub_set=5)
        self.RH_interDMA.plot()

    @property
    def mean_growth_factor(self):
        """Calculates the mean growthfactor of the particular size bin."""
        if '__mean_growth_factor' not in dir(self):


            def mean_linewise(gf_dist):
                growthfactors = self.hyg_distributions.data.minor_axis.values
                # meanl = ((gf_dist[~ gf_dist.mask] * np.log10(growthfactors[~ gf_dist.mask])).sum()/gf_dist[~gf_dist.mask].sum())
                meanl = ((gf_dist[~ np.isnan(gf_dist)] * np.log10(growthfactors[~ np.isnan(gf_dist)])).sum()/gf_dist[~np.isnan(gf_dist)].sum())
                stdl = np.sqrt((gf_dist[~ np.isnan(gf_dist)] * (np.log10(growthfactors[~ np.isnan(gf_dist)]) - meanl)**2).sum()/gf_dist[~np.isnan(gf_dist)].sum())
                return np.array([10**meanl,stdl])
            data = self.hyg_distributions.data
            allmeans = timeseries.TimeSeries_3D(pd.Panel(items=data.items, major_axis=data.major_axis, minor_axis= ['mean', 'std_log']))
            for i,time in enumerate(data.values):
                for e,size in enumerate(time):
                    allmeans.data.iloc[i,e] = mean_linewise(size)
            self.__mean_growth_factor = allmeans
        return self.__mean_growth_factor

def _concat_rules(arm_data_objs):
    out = ArmDatasetSub(False)
    out.RH_interDMA = timeseries.TimeSeries(pd.concat([i.RH_interDMA.data for i in arm_data_objs]))
    out.hyg_distributions = timeseries.TimeSeries_3D(pd.concat([i.hyg_distributions.data for i in arm_data_objs]))
    out.time_stamps = out.RH_interDMA.data.index
    return out
