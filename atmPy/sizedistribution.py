import numpy as np
from matplotlib.colors import LogNorm
import pylab as plt
from copy import deepcopy
import hagmods as hm
import pandas as pd
import warnings


distTypes = {'log normal': ['dNdlogDp','dSdlogDp','dVdlogDp'],
             'natural': ['dNdDp','dSdDp','dVdDp'],
             'number': ['dNdlogDp', 'dNdDp'],
             'surface': ['dSdlogDp','dSdDp'],
             'volume': ['dVdlogDp','dVdDp']}

def read_distribution_csv(fname, fixGaps = True):
    headerNo = 5
    rein = open(fname, 'r')
    nol = ['distributionType']
    outDict = {}
    for i in range(headerNo):
        split  = rein.readline().split('=')
        variable = split[0].strip()
        if split[0][0] == '#':
            break
        value = split[1].strip()
        if variable in nol:
            outDict[variable] = value
        else:
            outDict[variable] = np.array(eval(value))

    rein.close()
    data = pd.read_csv(fname, header=5, index_col=0)
    data.index = pd.to_datetime(data.index)
    distRein = aerosolSizeDistribution(data,outDict['bins'],outDict['distributionType'], fixGaps = fixGaps)
    return distRein

def get_label(distType):
    if distType =='dNdDp':
        label = '$\mathrm{d}N\,/\,\mathrm{d}D_{P}$ (nm$^{-1}\,$cm$^{-3}$)'     
    elif distType =='dNdlogDp':
        label = '$\mathrm{d}N\,/\,\mathrm{d}log(D_{P})$ (cm$^{-3}$)'
    elif distType =='dSdDp':
        label = '$\mathrm{d}S\,/\,\mathrm{d}D_{P}$ (nm$\,$cm$^{-3}$)'
    elif distType =='dSdlogDp':
        label = '$\mathrm{d}S\,/\,\mathrm{d}log(D_{P})$ (nm$^2\,$cm$^{-3}$)'  
    elif distType =='dVdDp':
        label = '$\mathrm{d}V\,/\,\mathrm{d}D_{P}$ (nm$^2\,$cm$^{-3}$)'
    elif distType =='dVdlogDp':
        label = '$\mathrm{d}V\,/\,\mathrm{d}log(D_{P})$ (nm$^3\,$cm$^{-3}$)'
    elif distType == 'calibration':
        label = '$\mathrm{d}N\,/\,\mathrm{d}Amp$ (bin$^{-1}\,$cm$^{-3}$)'
    elif distType == 'numberConcentration':
        label = 'Particle number in bin'
    else:
        raise ValueError('%s is not really an option!?!'%distType)
    return label

class aerosolSizeDistribution:
    """data: pandas dataFrame with 
                 - column names (each name is something like this: '150-200')
                 - index is time (at some point this should be arbitrary, convertable to altitude for example?)
       unit conventions:
             - diameters: nanometers
             - flowrates: cc (otherwise, axis label need to be adjusted an caution needs to be taken when dealing is AOD)
       distributionType:  
             log normal: 'dNdlogDp','dSdlogDp','dVdlogDp'
             natural: 'dNdDp','dSdDp','dVdDp'
             number: 'dNdlogDp', 'dNdDp'
             surface: 'dSdlogDp','dSdDp'
             volume: 'dVdlogDp','dVdDp'
       bincenters: this is if you actually want to pass the bincenters, if False they will be calculated """
       
    def __init__(self,data, bins, distributionType, bincenters = False, fixGaps = True):
        self.data = data
        self.bins = bins
        if type(bincenters) == np.ndarray:
            self.bincenters = bincenters
        else:
            self.bincenters = (bins[1:] + bins[:-1])/2.
        self.binwidth = (bins[1:] - bins[:-1])
        self.distributionType = distributionType
        if fixGaps:
            self.fillGapsWithZeros()
#        self.differentialStyle = differentialStyle #I don't think this is still used ... does not make much sence
            
    def fillGapsWithZeros(self,scale = 1.1):
        """Finds gaps in dataset (e.g. when instrument was shut of). 
        It adds one line of zeros to the beginning and one to the end of the gap. 
        Therefore the gap is visible as zeros instead of the interpolated values"""
        diff = self.data.index[1:].values - self.data.index[0:-1].values
        threshold = np.median(diff)*scale
        where = np.where(diff>threshold)[0]
        if len(where) != 0:
            warnings.warn('The dataset provided had %s gaps'%len(where))
            gap_start = self.data.index[where]
            gap_end = self.data.index[where+1]
            for gap_s in gap_start:
                self.data.loc[gap_s+threshold] = np.zeros(self.bincenters.shape)
            for gap_e in gap_end:
                self.data.loc[gap_e-threshold] = np.zeros(self.bincenters.shape)
            self.data = self.data.sort_index()
        return
        
    def _getXYZ(self):
        """ This will create three arrays, so when plotted with pcolor each pixel will represent the exact bin width"""
        binArray = np.repeat(np.array([self.bins]),self.data.index.shape[0], axis=0)
        timeArray = np.repeat(np.array([self.data.index.values]), self.bins.shape[0], axis = 0).transpose()
        ext = np.array([np.zeros(self.data.index.values.shape)]).transpose()
        Z =  np.append(self.data.values, ext, axis = 1)
        return timeArray,binArray,Z
        
    def plot_distribution(self, vmax = None, vmin = None, norm = 'linear',showMinorTickLabels = True, removeTickLabels = ["700", "900"], plotOnTheseAxes = False):
        """ plots and returns f,a,pc,cb (figure, axis, pcolormeshInstance, colorbar)
        Optional parameters:
              norm - ['log','linear']"""
        X,Y,Z = self._getXYZ()
        f,a = plt.subplots()
        if norm == 'log':
            norm = LogNorm()
        elif norm == 'linear':
            norm = None
        pc = a.pcolormesh( X,Y,Z,vmin= vmin, vmax = vmax, norm = norm, cmap = hm.get_colorMap_intensity())
        a.set_yscale('log')
        a.set_ylim((self.bins[0],self.bins[-1]))
        a.set_xlabel('Time (UTC)')
        if self.distributionType == 'claibration':
            a.set_ylabel('Amplitude (digitizer bins)')
        else:
            a.set_ylabel('Diameter (nm)')
        cb = f.colorbar(pc)
        label = get_label(self.distributionType)
        cb.set_label(label)
        
#        if self.distributionType =='dNdDp':
#            cb.set_label('$\mathrm{d}N\,/\,\mathrm{d}D_{P}$ (nm$^{-1}\,$cm$^{-3}$)')        
#        elif self.distributionType =='dNdlogDp':
#            cb.set_label('$\mathrm{d}N\,/\,\mathrm{d}log(D_{P})$ (cm$^{-3}$)')   
#        elif self.distributionType =='dSdDp':
#            cb.set_label('$\mathrm{d}S\,/\,\mathrm{d}D_{P}$ (nm$\,$cm$^{-3}$)')   
#        elif self.distributionType =='dSdlogDp':
#            cb.set_label('$\mathrm{d}S\,/\,\mathrm{d}log(D_{P})$ (nm$^2\,$cm$^{-3}$)')   
#        elif self.distributionType =='dVdDp':
#            cb.set_label('$\mathrm{d}V\,/\,\mathrm{d}D_{P}$ (nm$^2\,$cm$^{-3}$)')     
#        elif self.distributionType =='dVdlogDp':
#            cb.set_label('$\mathrm{d}V\,/\,\mathrm{d}log(D_{P})$ (nm$^3\,$cm$^{-3}$)')
#        elif self.distributionType == 'calibration':
#            cb.set_label('$\mathrm{d}N\,/\,\mathrm{d}Amp$ (bin$^{-1}\,$cm$^{-3}$)')
#        else:
#            raise ValueError('%s is not really an option!?!'%self.distributionType)
        
        f.autofmt_xdate()
        if self.distributionType != 'calibration':
            a.yaxis.set_minor_formatter(plt.FormatStrFormatter("%i"))
            a.yaxis.set_major_formatter(plt.FormatStrFormatter("%i"))
            
            f.canvas.draw() # this is important, otherwise the ticks (at least in case of minor ticks) are not created yet
    #        delList =
            ticks = a.yaxis.get_minor_ticks()
            for i in ticks:
                if i.label.get_text() in removeTickLabels:
                    i.label.set_visible(False)           
        return f,a,pc,cb
        
    def zoom_time(self, start = None, end = None):
        """'2014-11-24 16:02:30'"""
        dist = self.copy()
        dist.data = dist.data.truncate(before=start, after = end)
        return dist
                 

    def _normal2log(self):
        trans = (self.bincenters * np.log(10.))
        return trans
    
    def _2Surface(self):
        trans  = 4. * np.pi * (self.bincenters/2.)**2
        return trans
        
    def _2Volume(self):
        trans  = 4./3. * np.pi * (self.bincenters/2.)**3
        return trans
    
    def _convert2otherDistribution(self,distType):
        dist = self.copy()
        if dist.distributionType == distType:
            warnings.warn('Distribution type is already %s. Output is an unchanged copy of the distribution'%distType)
            return dist
        
        
        if dist.distributionType == 'numberConcentration':
            pass        
        elif distType == 'numberConcentration':
            pass    
        elif dist.distributionType in distTypes['log normal']:
            if distType in distTypes['log normal']:
                print 'both log normal'
            else:
                dist.data = dist.data / self._normal2log()
                
        elif dist.distributionType in distTypes['natural']:
            if distType in distTypes['natural']:
                print 'both natural'
            else:
                dist.data = dist.data * self._normal2log() 
        else:
            raise ValueError('%s is not an option'%distType)
            
        if dist.distributionType == 'numberConcentration':
            pass       
        
        elif distType == 'numberConcentration':
            pass                
        elif dist.distributionType in distTypes['number']:
            if distType in distTypes['number']:
                print 'both number'
            else:
                if distType in distTypes['surface']:
                    dist.data = dist.data * self._2Surface() 
                elif distType in distTypes['volume']:
                    dist.data = dist.data * self._2Volume()
                else:
                    raise ValueError('%s is not an option'%distType)

        elif dist.distributionType in distTypes['surface']:
            if distType in distTypes['surface']:
                print 'both surface'
            else:
                if distType in distTypes['number']:
                    dist.data = dist.data / self._2Surface() 
                elif distType in distTypes['volume']:
                    dist.data = dist.data * self._2Volume() / self._2Surface()  
                else:
                    raise ValueError('%s is not an option'%distType)             
                    
        elif dist.distributionType in distTypes['volume']:
            if distType in distTypes['volume']:
                print 'both volume'
            else:
                if distType in distTypes['number']:
                    dist.data = dist.data / self._2Volume()
                elif distType in distTypes['surface']:
                    dist.data = dist.data * self._2Surface()  / self._2Volume() 
                else:
                    raise ValueError('%s is not an option'%distType)
        else:
            raise ValueError('%s is not an option'%distType)     
            
            
        
        if distType == 'numberConcentration':
            dist = dist.convert2dNdDp()
            dist.data = dist.data * self.binwidth
            
        elif dist.distributionType == 'numberConcentration':
            dist.data = dist.data/self.binwidth
            dist.distributionType = 'dNdDp'
            dist = dist._convert2otherDistribution(distType)          
        
        dist.distributionType = distType
        print 'converted from %s to %s'%(self.distributionType,dist.distributionType)
        return dist
        
    def convert2dNdDp(self):
        return self._convert2otherDistribution('dNdDp')
        
    def convert2dNdlogDp(self):
        return self._convert2otherDistribution('dNdlogDp')
        
    def convert2dSdDp(self):
        return self._convert2otherDistribution('dSdDp')
        
    def convert2dSdlogDp(self):
        return self._convert2otherDistribution('dSdlogDp')
        
    def convert2dVdDp(self):
        return self._convert2otherDistribution('dVdDp')
        
    def convert2dVdlogDp(self):
        return self._convert2otherDistribution('dVdlogDp')
        
    def convert2numberconcentration(self):
        return self._convert2otherDistribution('numberConcentration')
                
        
#    def convert2LogNormal(self):
#        dist = self.copy()
#        if dist.differentialStyle == 'dNdLogDp':
#            warnings.warn('differential style is already in log normal. Simple copy is returned')
#        dist.data = dist.data * dist.bincenters * np.log(10.)
#        dist.differentialStyle = 'dNdLogDp'
#        return dist
#        
#    def convert2surfaceDistribution(self):
#        dist = self.copy()
#        if dist.distributionType == 'surface':
#            warnings.warn('is already surface distribution. Simple copy returned')
#        elif dist.distributionType != 'number':
#            dist = dist.convert2numberDistribution()
#        if dist.distributionType == 'number':
#            dist.data = dist.data * 4. * np.pi * (dist.bincenters/2.)**2
#        dist.distributionType = 'surface'
#        return dist
#    
#    
#    def convert2volumeDistribution(self):
#        dist = self.copy()
#        if dist.distributionType == 'volume':
#            warnings.warn('is already volume distribution. Simple copy returned')
#        elif dist.distributionType != 'number':
#            dist = dist.convert2numberDistribution()
#        if dist.distributionType == 'number':
#            dist.data = dist.data * 4./3. * np.pi * (dist.bincenters/2.)**3
#        dist.distributionType = 'volume'
#        return dist
#        
#    def convert2numberDistribution(self):
#        dist = self.copy()
#        if dist.distributionType == 'volume':
#            dist.data = dist.data / ( 4./3. * np.pi * (dist.bincenters/2.)**3)
#        elif dist.distributionType == 'surface':
#            dist.data = dist.data / ( 4. * np.pi * (dist.bincenters/2.)**2)
#        elif dist.distributionType == 'number':
#            warnings.warn('is already number distribution. Simple copy returned')
#        dist.distributionType = 'number'
#        return dist
        
    
    def average_overTime(self, window='1S'):
        dist = self.copy()
        window = window
        dist.data = dist.data.resample(window, closed='right',label='right')
        if dist.distributionType == 'calibration':
            dist.data.values[np.where(np.isnan(self.data.values))] = 0
        return dist
        
    def average_overAllTime(self):
        '''averages over the entire dataFrame and returns a single sizedistribution (numpy.ndarray)'''
        singleHist = np.zeros(self.data.shape[1])
        for i in xrange(self.data.shape[1]):
            line = self.data.values[:,i]
            singleHist[i] = np.average(line[~np.isnan(line)])
        return singleHist
        
    def copy(self):
        return deepcopy(self)
        
    def save_csv(self, fname):
        raus = open(fname, 'w')
        raus.write('bins = %s\n'% self.bins.tolist())
#        raus.write('bincenter = %s\n'% self.bincenters.tolist())
#        raus.write('binwidth = %s\n'% self.binwidth.tolist())
        raus.write('distributionType = %s\n'%self.distributionType)
        raus.write('#\n')
        raus.close()
        self.data.to_csv(fname, mode = 'a')
        return
        

        
class aerosolLayers:
    """data: pandas dataFrame with 
                 - column names (each name is something like this: '150-200')
                 - altitude (at some point this should be arbitrary, convertable to altitude for example?)
       unit conventions:
             - diameters: nanometers
             - flowrates: cc (otherwise, axis label need to be adjusted an caution needs to be taken when dealing is AOD) 
       distributionType:  
             log normal: 'dNdlogDp','dSdlogDp','dVdlogDp'
             natural: 'dNdDp','dSdDp','dVdDp'
             number: 'dNdlogDp', 'dNdDp'
             surface: 'dSdlogDp','dSdDp'
             volume: 'dVdlogDp','dVdDp'"""
    def __init__(self,data, bins, distributionType, layerBoundery):
        self.data = data
        self.bins = bins
        self.bincenters = (bins[1:] + bins[:-1])/2.
        self.binwidth = (bins[1:] - bins[:-1])
        self.distributionType = distributionType
        
        self.layerBoundery = layerBoundery
        self.layerThickness = layerBoundery[1:] - layerBoundery[:-1]
        self.layerCenter = (layerBoundery[1:] + layerBoundery[:-1])/2.
        
    def _getXYZ(self):
        """ This will create three arrays, so when plotted with pcolor each pixel will represent the exact bin width"""
        print 'need fixn'
        return False
#        binArray = np.repeat(np.array([self.bins]),self.data.index.shape[0], axis=0)
#        timeArray = np.repeat(np.array([self.data.index.values]), self.bins.shape[0], axis = 0).transpose()
#        ext = np.array([np.zeros(self.data.index.values.shape)]).transpose()
#        Z =  np.append(self.data.values, ext, axis = 1)
#        return timeArray,binArray,Z
    def plot_eachLayer(self, a = None):
        """ plots the distribytion of each layer in one plot"""
        if not a:
            f,a = plt.subplots()
        else:
            f = None
            pass
        for iv in self.data.index.values:
            a.plot(self.bincenters,self.data.loc[iv,:], label = '%i'%iv)
        a.set_xlabel('Particle diameter (nm)')
        a.set_ylabel(get_label(self.distributionType))
        a.legend()
        a.semilogx()
        return f,a
            
        
        
    def plot_distribution(self, removeTickLabels = ["700", "900"], plotOnTheseAxes = False):
        """ plots and returns f,a,pc,cb (figure, axis, pcolormeshInstance, colorbar)"""
        print 'need fixn'
        return False
#        X,Y,Z = self._getXYZ()
#        f,a = plt.subplots()
#        pc = a.pcolormesh(X,Y,Z, norm = LogNorm(), cmap = hm.get_colorMap_intensity())
#        a.set_yscale('log')
#        a.set_ylim((self.bins[0],self.bins[-1]))
#        a.set_xlabel('Time (UTC)')
#        if self.distributionType == 'claibration':
#            a.set_ylabel('Amplitude (digitizer bins)')
#        else:
#            a.set_ylabel('Diameter (nm)')
#        cb = f.colorbar(pc)
#        
#        if self.distributionType =='dNdDp':
#            cb.set_label('$\mathrm{d}N\,/\,\mathrm{d}D_{P}$ (cm$^{-3}$)')        
#        elif self.distributionType =='dNdlogDp':
#            cb.set_label('$\mathrm{d}N\,/\,\mathrm{d}log(D_{P})$ (cm$^{-3}$)')   
#        elif self.distributionType =='dSdDp':
#            cb.set_label('$\mathrm{d}S\,/\,\mathrm{d}D_{P}$ (nm$^2\,$cm$^{-3}$)')   
#        elif self.distributionType =='dSdlogDp':
#            cb.set_label('$\mathrm{d}S\,/\,\mathrm{d}log(D_{P})$ (nm$^2\,$cm$^{-3}$)')   
#        elif self.distributionType =='dVdDp':
#            cb.set_label('$\mathrm{d}V\,/\,\mathrm{d}D_{P}$ (nm$^2\,$cm$^{-3}$)')     
#        elif self.distributionType =='dVdlogDp':
#            cb.set_label('$\mathrm{d}V\,/\,\mathrm{d}log(D_{P})$ (nm$^2\,$cm$^{-3}$)')   
#        elif self.distributionType =='dSdlogDp':
#            cb.set_label('$\mathrm{d}N\,/\,\mathrm{d}log(D_{P})$ (cm$^{-3}$)')   
#        else:
#            raise ValueError('%s is not really an option!?!'%self.distributionType)
#        
#        f.autofmt_xdate()
#        a.yaxis.set_minor_formatter(plt.FormatStrFormatter("%i"))
#        a.yaxis.set_major_formatter(plt.FormatStrFormatter("%i"))
#        
#        f.canvas.draw() # this is important, otherwise the ticks (at least in case of minor ticks) are not created yet
##        delList =
#        ticks = a.yaxis.get_minor_ticks()
#        for i in ticks:
#            if i.label.get_text() in removeTickLabels:
#                i.label.set_visible(False)           
#        return f,a,pc,cb
        
    def zoom_altitude(self, start = None, end = None):
        """'2014-11-24 16:02:30'"""
        print 'need fixn'
        return False
#        dist = self.copy()
#        dist.data = dist.data.truncate(before=start, after = end)
#        return dist
#                 

    def _normal2log(self):
        trans = (self.bincenters * np.log(10.))
        return trans
    
    def _2Surface(self):
        trans  = 4. * np.pi * (self.bincenters/2.)**2
        return trans
        
    def _2Volume(self):
        trans  = 4./3. * np.pi * (self.bincenters/2.)**3
        return trans
    
    def _convert2otherDistribution(self,distType):
        dist = self.copy()
        if dist.distributionType == distType:
            warnings.warn('Distribution type is already %s. Output is an unchanged copy of the distribution'%distType)
            return dist
        
        
        if dist.distributionType == 'numberConcentration':
            pass        
        elif distType == 'numberConcentration':
            pass    
        elif dist.distributionType in distTypes['log normal']:
            if distType in distTypes['log normal']:
                print 'both log normal'
            else:
                dist.data = dist.data / self._normal2log()
                
        elif dist.distributionType in distTypes['natural']:
            if distType in distTypes['natural']:
                print 'both natural'
            else:
                dist.data = dist.data * self._normal2log() 
        else:
            raise ValueError('%s is not an option'%distType)
            
        if dist.distributionType == 'numberConcentration':
            pass       
        
        elif distType == 'numberConcentration':
            pass                
        elif dist.distributionType in distTypes['number']:
            if distType in distTypes['number']:
                print 'both number'
            else:
                if distType in distTypes['surface']:
                    dist.data = dist.data * self._2Surface() 
                elif distType in distTypes['volume']:
                    dist.data = dist.data * self._2Volume()
                else:
                    raise ValueError('%s is not an option'%distType)

        elif dist.distributionType in distTypes['surface']:
            if distType in distTypes['surface']:
                print 'both surface'
            else:
                if distType in distTypes['number']:
                    dist.data = dist.data / self._2Surface() 
                elif distType in distTypes['volume']:
                    dist.data = dist.data * self._2Volume() / self._2Surface()  
                else:
                    raise ValueError('%s is not an option'%distType)             
                    
        elif dist.distributionType in distTypes['volume']:
            if distType in distTypes['volume']:
                print 'both volume'
            else:
                if distType in distTypes['number']:
                    dist.data = dist.data / self._2Volume()
                elif distType in distTypes['surface']:
                    dist.data = dist.data * self._2Surface()  / self._2Volume() 
                else:
                    raise ValueError('%s is not an option'%distType)
        else:
            raise ValueError('%s is not an option'%distType)     
            
            
        
        if distType == 'numberConcentration':
            dist = dist.convert2dNdDp()
            dist.data = dist.data * self.binwidth
            
        elif dist.distributionType == 'numberConcentration':
            dist.data = dist.data/self.binwidth
            dist.distributionType = 'dNdDp'
            dist = dist._convert2otherDistribution(distType)          
        
        dist.distributionType = distType
        print 'converted from %s to %s'%(self.distributionType,dist.distributionType)
        return dist
  
    def convert2dNdDp(self):
        return self._convert2otherDistribution('dNdDp')
        
    def convert2dNdlogDp(self):
        return self._convert2otherDistribution('dNdlogDp')
        
    def convert2dSdDp(self):
        return self._convert2otherDistribution('dSdDp')
        
    def convert2dSdlogDp(self):
        return self._convert2otherDistribution('dSdlogDp')
        
    def convert2dVdDp(self):
        return self._convert2otherDistribution('dVdDp')
        
    def convert2dVdlogDp(self):
        return self._convert2otherDistribution('dVdlogDp')
                
    def convert2numberconcentration(self):
        return self._convert2otherDistribution('numberConcentration')
        
    
    def average_overAltitude(self, window='1S'):
        print 'need fixn'
        return False
#        window = window
#        self.data = self.data.resample(window, closed='right',label='right')
#        if self.distributionType == 'calibration':
#            self.data.values[np.where(np.isnan(self.data.values))] = 0
#        return
        
    def average_overAllAltitudes(self):
        print 'need fixn'
        return False
#        singleHist = np.zeros(self.data.shape[1])
#        for i in xrange(self.data.shape[1]):
#            line = self.data.values[:,i]
#            singleHist[i] = np.average(line[~np.isnan(line)])
#        return singleHist
        
    def copy(self):
        return deepcopy(self)
        
    def save_csv(self, fname):
        raus = open(fname, 'w')
        raus.write('bins = %s\n'% self.bins.tolist())
        raus.write('bincenter = %s\n'% self.bincenters.tolist())
        raus.write('binwidth = %s\n'% self.binwidth.tolist())
        raus.write('distributionType = %s\n'%self.distributionType)
        raus.write('differentialStyle = %s\n'% self.differentialStyle)
        raus.write('\n')
        raus.close()
        self.data.to_csv(fname, mode = 'a')
        return
        
def generate_aerosolLayer(diameter = [.01,2.5],
                                 numberOfDiameters = 30,
                                 centerOfAerosolMode = 0.6,
                                 widthOfAerosolMode = 0.2,
                                 numberOfParticsInMode = 10000,
                                 layerBoundery = [0.,10000],
                                 ):
    """generates a numberconcentration of an aerosol layer which has a gaussian shape when plottet in dN/log(Dp). 
    However, returned is a numberconcentrations (simply the number of particles in each bin, no normalization)
    Returns
        Number concentration (#)
        bin edges (nm)"""
        
    
    layerBoundery = np.array(layerBoundery)    
    start = diameter[0]
    end = diameter[1]
    noOfD = numberOfDiameters
    centerDiameter = centerOfAerosolMode
    width = widthOfAerosolMode
    bins = np.linspace(np.log10(start),np.log10(end),noOfD)
    binwidth = bins[1:]-bins[:-1]
    bincenters= (bins[1:] + bins[:-1])/2.
    dNDlogDp = plt.mlab.normpdf(bincenters,np.log10(centerDiameter),width)
    extraScale = 1
    scale = 1
    while 1:
        NumberConcent = dNDlogDp*binwidth*scale*extraScale
        if scale!=1:
            break
        else:            
            scale = float(numberOfParticsInMode)/NumberConcent.sum() 
            
    binEdges = 10**bins
#    diameterBinCenters = (binEdges[1:] + binEdges[:-1])/2.
    diameterBinwidth = binEdges[1:] - binEdges[:-1]
    
    cols = []
    for e,i in enumerate(binEdges[:-1]):
            cols.append(str(i)+'-'+str(binEdges[e+1]))
    
    layerBoundery = np.array([0.,10000.])
#    layerThickness = layerBoundery[1:] - layerBoundery[:-1]
    layerCenter = [5000.]
    data = pd.DataFrame(np.array([NumberConcent/diameterBinwidth]),index = layerCenter , columns=cols)
#     return data
    
#     atmosAerosolNumberConcentration = pd.DataFrame()
#     atmosAerosolNumberConcentration['bin_center'] = pd.Series(diameterBinCenters)
#     atmosAerosolNumberConcentration['bin_start'] = pd.Series(binEdges[:-1])
#     atmosAerosolNumberConcentration['bin_end'] = pd.Series(binEdges[1:])
#     atmosAerosolNumberConcentration['numberConcentration'] = pd.Series(NumberConcent)
#     return atmosAerosolNumberConcentration

    return aerosolLayers(data, binEdges,'dNdDp',layerBoundery)
    
def test_generate_numberConcentration():
    """result should look identical to Atmospheric Chemistry and Physis page 422"""
    nc = generate_aerosolLayer(diameter=[0.01, 10],
                                     centerOfAerosolMode= 0.8,
                                     widthOfAerosolMode=0.3,
                                     numberOfDiameters=100,
                                     numberOfParticsInMode = 1000,
                                     layerBoundery=[0.0, 10000]
                                     )

    plt.plot(nc.bincenters,nc.data.values[0].transpose()*nc.binwidth, label = 'numberConc')
    plt.plot(nc.bincenters,nc.data.values[0].transpose(), label = 'numberDist')
    ncLN = nc.convert2dNdlogDp()
    plt.plot(ncLN.bincenters,ncLN.data.values[0].transpose(), label = 'LogNormal')
    plt.legend()
    plt.semilogx()            