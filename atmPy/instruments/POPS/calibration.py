#from POPS_lib.fileIO import read_Calibration_fromFile,read_Calibration_fromString,save_Calibration
#import fileIO
from scipy.interpolate import UnivariateSpline
import numpy as np
import pylab as plt
# from StringIO import StringIO as io
import pandas as pd
import warnings

#read_fromFile = fileIO.read_Calibration_fromFile
#read_fromString = fileIO.read_Calibration_fromString

def _string2Dataframe(data):
    sb = io(data)
    dataFrame = pd.read_csv(sb, sep = ' ', names = ('d','amp')).sort('d')
    return dataFrame

def read_Calibration_fromString(data):
    '''unit of diameter must be nm
    e.g.:
data = """140 88
150 102
173 175
200 295
233 480
270 740
315 880
365 1130
420 1350
490 1930
570 3050
660 4200
770 5100
890 6300
1040 8000
1200 8300
1400 10000
1600 11500
1880 16000
2180 21000
2500 28000
3000 37000"""
    '''
    
    dataFrame = _string2Dataframe(data)
    calibrationInstance = calibration(dataFrame)
    return calibrationInstance

def read_Calibration_fromFile(fname):
    """ most likely found here"""
    calDataFrame = pd.read_csv(fname)
    calibrationInstance = calibration(calDataFrame)
    return calibrationInstance

def save_Calibration(calibrationInstance, fname):
    """should be saved hier cd ~/data/POPS_calibrations/"""
    calibrationInstance.data.to_csv(fname, index = False)
    return

class calibration:
    def __init__(self,dataTabel):
        self.data = dataTabel
        self.calibrationFunction = self.get_calibrationFunctionSpline()
        
        
    def save_csv(self,fname):
        save_Calibration(self,fname)
        return
        
    def get_calibrationFunctionSpline(self, fitOrder = 1):# = 1, noOfPts = 500, plot = False):
        """
        Performes a spline fit/smoothening (scipy.interpolate.UnivariateSpline) of d over amp (yes this way not the other way around).
    
        Returns (generates): creates a function self.spline which can later be used to calculate d from amp 
    
        Optional Parameters:
        \t s: int - oder of the spline function
        \t noOfPts: int - length of generated graph
        \t plot: boolean - if result is supposed to be plotted
        """       
        # The following two step method is necessary to get a smooth curve. 
        #When I only do the second step on the cal_curve I get some wired whiggles
        ##### First Step
        if (self.data.amp.values[1:]-self.data.amp.values[:-1]).min() < 0:
            warnings.warn('The data represent a non injective function! This will not work. plot the calibration to see what I meen')  
        
        fitOrder = 1
        sf = UnivariateSpline(self.data.d.values, self.data.amp.values, s=fitOrder)
        d = np.logspace(np.log10(self.data.d.values.min()), np.log10(self.data.d.values.max()), 500)
        amp = sf(d)
    
        ##### second step
        cal_function = UnivariateSpline(amp, d, s=fitOrder)
        return cal_function
        
    def plot_calibration(self):
        """Plots the calibration function and data
        Arguments
        ------------
            cal: calibration instance
        
        Returns
        ------------
            figure
            axes
            calibration data graph
            calibration function graph
        """
        cal_function = self.calibrationFunction
        amp = np.logspace(np.log10(self.data.amp.min()), np.log10(self.data.amp.max()), 500)
        d = cal_function(amp)
    
        f,a = plt.subplots()
        
        cal_data, = a.plot(self.data.d,  self.data.amp, 'o',label = 'data',)
        cal_func, = a.plot(d,amp, label = 'function')
        
        a.loglog()
    
        a.set_xlim(0.9*self.data.d.min(), 1.1*self.data.d.max())
        a.set_xlabel('Diameter (nm)')#($\mu$m)')
    
        a.set_ylim(0.9*self.data.amp.min(), 1.1*self.data.amp.max()) 
        a.set_ylabel('Amplitude (digitizer bins)')
    
        a.set_title('Calibration curve')
        a.legend(loc = 2)
        return f,a,cal_data, cal_func