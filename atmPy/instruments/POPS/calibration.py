#from POPS_lib.fileIO import read_Calibration_fromFile,read_Calibration_fromString,save_Calibration
#import fileIO
from scipy.interpolate import UnivariateSpline
import numpy as np
import pylab as plt
from io import StringIO as io
import pandas as pd
import warnings

#read_fromFile = fileIO.read_Calibration_fromFile
#read_fromString = fileIO.read_Calibration_fromString

def _msg(txt, save, out_file, verbose):
    if verbose:
        print(txt)
    if save:
        out_file.write(str(txt) + '\n')


def get_interface_bins(fname, n_bins, imin=1.4, imax=4.8, save=False, verbose = True):
    """Prints the bins assosiated with what is seen on the POPS user interface and the serial output, respectively.

    Parameters
    ----------
    fname: string or calibration instance
        name of file containing a calibration or a calibration instance it self
    n_bins: int
        number of bins
    imin: float [1.4], optional
        log10 of the minimum value considered (digitizer bins)
    imax: float [4.8], optional
        log10 of the maximum value considered (digitizer bins)
    save: bool or string.
        If result is saved into file given by string.


    Returns
    -------
    matplotlib axes instance
    pandas DataFrame instance
    """
    if isinstance(fname, str):
        cal = read_csv(fname)
    else:
        cal = fname

    bin_ed = np.linspace(imin, imax, n_bins + 1)
    bin_center_log = 10 ** ((bin_ed[:-1] + bin_ed[1:]) / 2.)
    bin_center_lin = ((10 ** bin_ed[:-1] + 10 ** bin_ed[1:]) / 2.)
    bin_ed = 10 ** bin_ed
    bin_ed_cal = cal.calibrationFunction(bin_ed)
    bin_center_lin_cal = cal.calibrationFunction(bin_center_lin)
    bin_center_log_cal = cal.calibrationFunction(bin_center_log)
    if save:
        save_file = open(save, 'w')
    else:
        save_file = False

    txt = '''
bin edges (digitizer bins)
--------------------------'''
    _msg(txt, save, save_file, verbose)

    for e, i in enumerate(bin_ed):
        _msg(i, save, save_file, verbose)
    # bin_center_cal = cal.calibrationFunction(bin_center)


    txt = '''
bin centers (digitizer bins)
----------------------------'''
    _msg(txt, save, save_file, verbose)
    for e, i in enumerate(bin_center_lin):
        _msg(i, save, save_file, verbose)

    txt = '''
bin centers of logarithms (digitizer bins)
----------------------------'''
    _msg(txt, save, save_file, verbose)
    for e, i in enumerate(bin_center_log):
        _msg(i, save, save_file, verbose)

    txt = '''

bin edges (nm)
--------------'''
    _msg(txt, save, save_file, verbose)
    for e, i in enumerate(bin_ed_cal):
        _msg(i, save, save_file, verbose)
    # bin_center_cal = cal.calibrationFunction(bin_center)


    txt = '''
bin centers (nm)
----------------'''
    _msg(txt, save, save_file, verbose)
    for e, i in enumerate(bin_center_lin_cal):
        _msg(i, save, save_file, verbose)

    txt = '''
bin centers of logarithms (nm)
----------------'''
    _msg(txt, save, save_file, verbose)
    for e, i in enumerate(bin_center_log_cal):
        _msg(i, save, save_file, verbose)

    out = {}

    df_bin_c = pd.DataFrame(bin_center_lin_cal, index=bin_center_log, columns=['Bin_centers'])
    df_bin_e = pd.DataFrame(bin_ed_cal, index = bin_ed, columns = ['Bin_edges'])
    # a = df.Bin_centers.plot()

    if verbose:
        f, a = plt.subplots()
        d = df_bin_c.Bin_centers.values[1:-1]
        g, = a.plot(np.arange(len(d)) + 2, d)
        g.set_linestyle('')
        g.set_marker('o')
        # g.set_label('')
        a.set_yscale('log')
        a.set_xlim((1, 16))
        a.set_ylim((100, 3000))
        a.set_ylabel('Bin center (nm)')
        a.grid(which='both')
        a.set_xlabel('POPS bin')
        out['axes'] = a
    else:
        out['axes'] = None

    # a.set_title('Bin')


    out['bincenters_v_int'] = df_bin_c
    out['binedges_v_int'] = df_bin_e
    return out


def _string2Dataframe(data, log=True):
    sb = io(data)
    dataFrame = pd.read_csv(sb, sep = ' ', names = ('d','amp')).sort('d')
    if log:
        dataFrame.amp = 10 ** dataFrame.amp
    return dataFrame


def read_str(data, log=True):
    '''Read a calibration table from string.

    Arguments
    ---------
    data: string.
        Multiline string with a diameter-intensity pair seperated by space. Diameter in nm, intensity in digitizer bin
        or log_10(digitizer bins).
    log: bool, optional.
        Set True if the intensity values are given in log_10(digitizer bins).

    Example
    -------
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
    2500 28000s
    3000 37000"""
    read_str(data, log = False)
    '''

    dataFrame = _string2Dataframe(data, log=log)
    calibrationInstance = calibration(dataFrame)
    return calibrationInstance


def read_csv(fname):
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
        
    def get_interface_bins(self, n_bins, imin=1.4, imax=4.8, save=False, verbose = False):
        out = get_interface_bins(self, n_bins, imin=imin, imax=imax, save=save, verbose = verbose)
        return out

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