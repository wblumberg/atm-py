import datetime
import os
import warnings
from struct import unpack, calcsize

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from atmPy.aerosols.size_distribution import sizedistribution
from atmPy.tools import miscell_tools as misc
from atmPy.general import timeseries as _timeseries

import pathlib
#from StringIO import StringIO as io
#from POPS_lib import calibration

#defaultBins = np.array([0.15, 0.168, 0.188, 0.211, 0.236, 0.264, 0.296, 0.332, 0.371, 0.416, 0.466, 0.522, 0.584, 0.655, 0.864, 1.14, 1.505, 1.987, 2.623, 3.462])
defaultBins = np.logspace(np.log10(140), np.log10(3000), 30)


#######
#### Peak file
def _read_PeakFile_Binary(fname, version = 'current', time_shift=0, skip_bites = 20, verbose = False):
    """returns a peak instance
    test_data_folder: ..."""
    assert(type(fname).__name__ == 'PosixPath')
    if version == 'current':
        version = 'BBB'

    if verbose:
        print('_read_PeakFile_Binary (version = {})'.format(version))
    directory, filename = os.path.split(fname)
    if version == 'labview':
        data = _binary2array_labview_clusters(fname, skip = skip_bites)
        if not np.any(data):
            return False
        dataFrame = _PeakFileArray2dataFrame(data,filename, time_shift, log = False, since_midnight = False)
    elif version == '01':
        data = _BinaryFile2Array(fname)
        dataFrame = _PeakFileArray2dataFrame(data,filename,time_shift)
    elif version == 'BBB': # Beaglebone system running POPS_BBB.c
        data = _bbb_binary2array(fname, 1, verbose = verbose)
        dataFrame = _PeakFileArray2dataFrame(data, fname, time_shift,
                                             log=False,
                                             since_midnight=False,
                                             BBBtype=1,
                                             verbose = verbose)
    elif version == 'BBB_dt': # Beaglebone system running POPS_BBB_dt.c
        data = _bbb_binary2array(fname, 2)
        dataFrame = _PeakFileArray2dataFrame(data, filename, time_shift,
                                             log=False,
                                             since_midnight=False,
                                             BBBtype=2)
    else:
        txt = 'This version does not exist: '%version
        raise ValueError(txt)

    peakInstance = peaks(dataFrame)
    return peakInstance


def read_binary(fname, pattern='Peak', time_shift = False ,version = 'current', ignore_error = False, skip_bites= 20, verbose = False):
    """Generates a single Peak instance from a file or list of files

    Arguments
    ---------
    fname: string or list of strings
    time_shift: iterable
        e.g. (1,'h)
        see http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units
    version: str
        'current' - current :-)
        '01': before summer-fall 2015
        'BBB': added by Gavin McMeeking to conver files produced by POPS_BBB.c Beaglebone version
        'BBB_dt': added by Gavin McMeeking to convert files produced by POPS_BBB_dt.c version 
    """

    m = None

    if type(fname) == str:
        # if os.path.isdir(fname):
            # fname = os.listdir(fname)
        fname = pathlib.Path(fname)

    elif type(fname).__name__ == 'PosixPath':
        if fname.is_file():
            if verbose:
                print('fname is file')
            pass
        elif fname.is_dir():
            if verbose:
                print('fname is folder', end = ' ... ')
            fname = list(fname.glob('*{}*'.format(pattern)))

    if type(fname).__name__ == 'list':
        if len(fname) == 0:
            raise ValueError('There are no files to be processed!')
        first = True
        for file in fname:
            if pattern not in file.name: # changed by GM because BBB file names slightly different than sbRIO
                print('%s is not a peak file ... skipped' % file)
                continue
            if verbose:
                print('%s ... processed' % file)

            mt = _read_PeakFile_Binary(file, version = version, time_shift=time_shift, skip_bites=skip_bites)

            # skipping if above returned False and ignore_error = True
            if not mt:
                if ignore_error:
                    txt = 'An error accured while trying to read file. File skipped!'
                    print(txt)
                    continue
                else:
                    txt = 'An error accured while trying to read file. Set ignore_error to True if you want this file tobe ignored.'
                    raise  ValueError(txt)

            if first:
                m = mt
                first = False
            else:
                m.data = pd.concat((m.data, mt.data))

    else:
        m = _read_PeakFile_Binary(fname, version = version, time_shift=time_shift, skip_bites=skip_bites, verbose = verbose)

    return m


#############################################
#############################################
#############################################
#############################################

# def _PeakFileArray2dataFrame(data,test_data_folder,deltaTime):
#     data = data.copy()
#     dateString = test_data_folder.split('_')[0]
#     dt = datetime.datetime.strptime(dateString, "%Y%m%d") - datetime.datetime.strptime('19700101', "%Y%m%d")
#     dts = dt.total_seconds()
#     #dtsPlus = datetime.timedelta(seconds = deltaTime).total_seconds()
#
#     columns = np.array(['Ticks', 'Amplitude', 'Width', 'Saturated', 'Masked'])
#
#
#     try:
#         Time_s = data[:,0]
#         rest = data[:,1:]
#         dataTable = pd.DataFrame(rest, columns=columns)
#         dataTable.index = pd.Series(pd.to_datetime(Time_s + dts + deltaTime, unit = 's'), name = 'Time_UTC')
#     except OverflowError:
#
#         data, report = _cleanPeaksArray(data)
#         warnings.warn('Binary file %s is corrupt. Will try to fix it. if no exception accured it probably worked\nReport:\n%s'%(test_data_folder,report))
#
#
#         Time_s = data[:,0]
#         rest = data[:,1:]
#         dataTable = pd.DataFrame(rest, columns=columns)
#         dataTable.index = pd.Series(pd.to_datetime(Time_s + dts + deltaTime, unit = 's'), name = 'Time_UTC')
#
#
#     dataTable.Amplitude = 10**dataTable.Amplitude # data is written in log10
#
#     dataTable.Ticks = dataTable.Ticks.astype(np.int32)
#     dataTable.Width = dataTable.Width.astype(np.int16)
#     dataTable.Saturated = dataTable.Saturated.astype(np.int16)
#     dataTable.Masked = np.abs(1. - dataTable.Masked).astype(np.int8)
#     return dataTable

def _csv2array(fname):
    df = pd.read_csv(fname).dropna()
    data = df.values
    return data


def _read_peak_file_csv(fname, deltaTime=0, log = True, since_midnight = True):
    """returns a peak instance
    test_data_folder: ...
    deltaTime: if you want to apply a timedelay in seconds"""
    data = _csv2array(fname)
    directory, fname = os.path.split(fname)
    dataFrame = _PeakFileArray2dataFrame(data, fname, deltaTime, log = log, since_midnight = since_midnight)
    peakInstance = peaks(dataFrame)
    return peakInstance


def read_csv(fname, log = True, since_midnight = True, verbose = False):
    """Generates a single Peak instance from a file or list of files

    Arguments
    ---------
    fname: string or list of strings
    log: bool.
        If the actual or the log of the amplitude is given. If log is True the value in the file will be 10**amp
    since_midnight: bool
        if the time stamp in the file is seconds since midnitght or since 19040101.
    """

    m = None
    if type(fname).__name__ == 'list':
        first = True
        for file in fname:
            if ('Peak.txt' not in file) and ('Peak.csv' not in file):
                if verbose:
                    print('%s is not a peak file ... skipped' % file)
                continue
            if verbose:
                print('%s ... processed' % file)
            mt = _read_peak_file_csv(file, log = log, since_midnight = since_midnight)
            if first:
                m = mt
                first = False
            else:
                m.data = pd.concat((m.data, mt.data))

    else:
        m = _read_peak_file_csv(fname, log = log, since_midnight = since_midnight)

    return m


#############################################
#############################################
#############################################
#############################################



def read_cal_process_peakFile(fname, cal, bins, average_over_time=False, normalize = False):
    """short cut to read, calibrate and furter process the peak file data
    Arguments
    ---------
    fname: str
        filename
    cal: calibration instance
    bins: array like
        bin-edges for binning of peak data to sizedistributions
    average_over_time: bool or str
        Downsampling of data, e.g. '60S' will downsample to 60 seconds
    normalize: float
        data will be divided by this number

    Returns
    -------
    size_dist_TS instance
    """

    peakdf = read_binary(fname)
    peakdf.apply_calibration(cal)
    dist = peakdf.peak2numberdistribution(bins=bins)
    if average_over_time:
        dist = dist.average_time(average_over_time)
    if normalize:
        dist.data *= 1./normalize
    dist = dist.convert2dNdlogDp()
    return dist


def _BinaryFile2Array(fname):
    entry_format = '>fLfBB?'
    field_names = np.array(['timeSincMitnight', 'ticks', 'log10_amplitude', 'width', 'saturated', 'use'])
    entry_size = calcsize(entry_format)

    rein = open(fname, mode='rb')
    entry_count = int(os.fstat(rein.fileno()).st_size / entry_size)

    data = np.zeros((entry_count , field_names.shape[0]))

    for i in range(entry_count):
        record = rein.read(entry_size)
        entry = np.array(unpack(entry_format, record))
        data[i] = entry
    rein.close()
    return data

def _bbb_binary2array(fname, bbbtype, verbose = False):
    if verbose:
        print('_bbb_binary2array', end = ' ... ')
    def read_time(file, entry_format='<d'):
        entry_size = calcsize(entry_format)
        record = file.read(entry_size)
        et = unpack(entry_format, record)[0]
        timet = et
        return timet

    def read_array_length(file, entry_format='<I'):
        entry_size = calcsize(entry_format)
        record = file.read(entry_size)
        lengtht = unpack(entry_format, record)[0]
        return lengtht

    def read_array(file, length, time, type):
        if type == 1:
            entry_format = '<IIII'
            ncol = 5
        elif type == 2:
            entry_format = '<III'
            ncol = 4

        entry_size = calcsize(entry_format)
        thearray = np.zeros((length, ncol))  # change based on BBB version

        for i in range(length):
            record = rein.read(entry_size)
            entry = unpack(entry_format, record)
            thearray[i, 0] = time
            thearray[i, 1:] = entry
        return thearray

    rein = open(fname, mode='rb')
    array_list = []
    while 1:
        try:
            length = read_array_length(rein)
            time = read_time(rein)
            array = read_array(rein, length, time, bbbtype)
            array_list.append(array)
        except:
            break

    full_array = np.concatenate(array_list)
    rein.close()
    if verbose:
        print('done')
    return full_array

def _binary2array_labview_clusters(fname, skip = 20):

    def read_time(file, entry_format = '>QQ'):
        entry_size = calcsize(entry_format)
        record = file.read(entry_size)
        et = unpack(entry_format, record)
        subsec = et[1] * 2**-64
        timet = et[0] + subsec
        return timet

    def read_array_length(file, entry_format = '>i'):
        entry_size = calcsize(entry_format)
        record = file.read(entry_size)
        lengtht  = unpack(entry_format, record)[0]
        return lengtht

    def read_array(rein, length, time, entry_format = '>LHBBB'):
        entry_size = calcsize(entry_format)

        thearray = np.zeros((length,6))

        for i in range(length):
            record = rein.read(entry_size)
            entry = unpack(entry_format, record)
            thearray[i,0] = time
            thearray[i,1:] = entry
        return thearray

    while 1:
        wrong_skip = False
        rein = open(fname, mode='rb')
        rein.read(skip) # This is some type of header ... no idea what exactly
        array_list = []
        while 1:
            try:
                time = read_time(rein)
                length = read_array_length(rein)
                array = read_array(rein, length,time)
                array_list.append(array)
            except:
                break

            lc = array[:,-1]

            # If a peak file was created on startup it will have a different header
            # length compared to when it was created because the maximum file size
            # was reached. The following test if the structure was correct an will
            # adjust the header length if necessary.
            if np.any(np.logical_and(lc != 1, lc != 0)):
                if skip == 0:
                    txt = "Sorry, this should not happen ... need fixn!!"
                    # raise ValueError(txt)
                    warnings.warn(txt)
                    return False
                wrong_skip = True
                skip = 0
                break

        rein.close()

        if wrong_skip:
            continue
        else:
            # looks like for short files the above skip test does not work ... quickfix:
            try:
                full_array = np.concatenate(array_list)
            except ValueError:
                skip = 0
                continue
            break

    return full_array


def _PeakFileArray2dataFrame(data,fname,time_shift, BBBtype = 0, log = True, since_midnight = True, verbose = False):
    if verbose:
        print(fname)
    assert(type(fname).__name__ == 'PosixPath')
    if verbose:
        print('_PeakFileArray2dataFrame', end = ' ... ')
    data = data.copy()

    # GRM added to deal with different binary file naming format between BBB and sbRIO
    if BBBtype == 0:
        dateString = fname.name.split('_')[0]
    else:
        pass
        # dateString = fname.split('_')[1][0:8]

    if since_midnight and BBBtype == 0:
        dt = datetime.datetime.strptime(dateString, "%Y%m%d") - datetime.datetime.strptime('19700101', "%Y%m%d")
        dts = dt.total_seconds()
    elif BBBtype == 0:
        dt = datetime.datetime.strptime('19040101', "%Y%m%d") - datetime.datetime.strptime('19700101', "%Y%m%d")
        dts = dt.total_seconds()
    else:
        dts = 0 # no time adjustment needed with BBB version?

    #dtsPlus = datetime.timedelta(seconds = deltaTime).total_seconds() 
    
    if BBBtype == 0:
        columns = np.array(['Ticks', 'Amplitude', 'Width', 'Saturated', 'Masked'])
    elif BBBtype == 1:
        columns = np.array(['Amplitude', 'Max', 'Width', 'Saturated'])
    elif BBBtype == 2:
        columns = np.array(['Max','Width','dt'])

    if time_shift:
        time_shift_in_sec = np.timedelta64(*time_shift)/np.timedelta64(1,'s')
    else:
        time_shift_in_sec = 0

    try:
        Time_s = data[:,0]
        rest = data[:,1:]
        dataTable = pd.DataFrame(rest, columns=columns)
        dataTable.index = pd.Series(pd.to_datetime(Time_s + dts + time_shift_in_sec, unit = 's'), name = 'Time_UTC')

    # corrupt file
    except pd.tslib.OutOfBoundsDatetime:
        Time_s = data[:, 0]
        threshold = 48*60*60
        txt = ('Something is wrong with the time stamp. Binary file %s is probably corrupt. All data after the first'
               'occurence of a suspect Timestamp will be ignored.')

        warnings.warn(txt)
        # Time_s[Time_s > threshold] =  np.nan
        point_of_termination = np.where(Time_s > threshold)[0][0]
        Time_s = Time_s[:point_of_termination]

        rest = data[:, 1:]
        rest = rest[:point_of_termination]

        dataTable = pd.DataFrame(rest, columns=columns)
        dataTable.index = pd.Series(pd.to_datetime(Time_s + dts + time_shift_in_sec, unit='s'), name='Time_UTC')

    # corrupt file, somewhat different though
    except OverflowError:
        data, report = _cleanPeaksArray(data)
        warnings.warn('Binary file %s is corrupt. Will try to fix it. if no exception accured it probably worked\nReport:\n%s'%(fname,report))
        
        
        Time_s = data[:,0]
        rest = data[:,1:]
        dataTable = pd.DataFrame(rest, columns=columns)
        dataTable.index = pd.Series(pd.to_datetime(Time_s + dts + time_shift_in_sec, unit = 's'), name = 'Time_UTC')
        
    if log:
        dataTable.Amplitude = 10**dataTable.Amplitude # data is written in log10
    if BBBtype == 0:
        dataTable.Ticks = dataTable.Ticks.astype(np.int32)
        dataTable.Width = dataTable.Width.astype(np.int16)
        dataTable.Saturated = dataTable.Saturated.astype(np.int16)
        dataTable.Masked = np.abs(1. - dataTable.Masked).astype(np.int8)
    elif BBBtype == 1:
        dataTable.Width = dataTable.Width.astype(np.int16)
    elif BBBtype == 2:
        dataTable.Width = dataTable.Width.astype(np.int16)
    if verbose:
        print('done')
    return dataTable

def _cleanPeaksArray(PeakArray):
    """tries to remove data points where obviously something went wrong. Returns the cleaned array."""
    BarrayClean = PeakArray.copy()
    startShape = BarrayClean.shape
    startstartShape = BarrayClean.shape

#    print BarrayClean.shape
    Tmax = 1.e6 #unless you are measuring for more than 2 weeks this should be ok
    BarrayClean = BarrayClean[BarrayClean[:,0] < Tmax]

    pointsRem = startShape[0] - BarrayClean.shape[0]
    report = '%s (%.5f%%) datapoints removed due to bad Time (quickceck)\n'%(pointsRem, pointsRem/float(startShape[0]))
    startShape = BarrayClean.shape
    ampMax = 2.*16 #the maximum you can measure with a 16 bit A2D converter
    BarrayClean = BarrayClean[BarrayClean[:,2] < ampMax]
    BarrayClean = BarrayClean[BarrayClean[:,2] > 0]

    pointsRem = startShape[0] - BarrayClean.shape[0]
    report += '%s (%.5f%%) datapoints removed due to bad Amplitude.\n'%(pointsRem, pointsRem/float(startShape[0]))
    startShape = BarrayClean.shape
    BarrayClean = BarrayClean[np.logical_or(BarrayClean[:,-1] == 1, BarrayClean[:,-1] == 0)]

    pointsRem = startShape[0] - BarrayClean.shape[0]
    report += '%s (%.5f%%) datapoints removed due to bad Used.\n'%(pointsRem, pointsRem/float(startShape[0]))
    startShape = BarrayClean.shape
    BarrayClean = BarrayClean[BarrayClean[:,3] < 1000]
    BarrayClean = BarrayClean[BarrayClean[:,3] > 1]

    pointsRem = startShape[0] - BarrayClean.shape[0]
    report +='%s (%.5f%%) datapoints removed due to bad Width.\n'%(pointsRem, pointsRem/float(startShape[0]))
    startShape = BarrayClean.shape
    BarUni = np.unique(BarrayClean[:,0])
    BarUniInt = BarUni[1:]- BarUni[:-1]
    timeMed = np.median(BarUniInt)

    lastTime = BarrayClean[0,0] #first entry in the time
    counterToHigh = 0
    counterToLow = 0
    for e,i in enumerate(BarrayClean):
        if i[0] - lastTime > timeMed * 1.1:
            i[0] = np.nan
            counterToHigh += 1
        elif i[0] - lastTime < 0:
            i[0] = np.nan
            counterToLow += 1
        else:
            lastTime = i[0]

    BarrayClean = BarrayClean[~np.isnan(BarrayClean[:,0])]


    pointsRem = startShape[0] - BarrayClean.shape[0]
    report += '%s (%.5f%%) datapoints removed due to bad Time (more elaborate check).\n'%(pointsRem, pointsRem/float(startShape[0]))
    startShape = BarrayClean.shape
    pointsRem = startstartShape[0] - BarrayClean.shape[0]
    report += 'All together %s (%.5f%%) datapoints removed.'%(pointsRem, pointsRem/float(startstartShape[0]))
    return BarrayClean, report

class peaks(object):
    def __init__(self,dataFrame):
        self.data = dataFrame

        # the Beaglebone code does not provide a "masked" field -> create a blank one
        if 'Masked' not in self.data.columns:
            self.data['Masked'] = 0
        
    def apply_calibration(self,calibrationInstance, verbose = False):
        self.data['Diameter'] = pd.Series(calibrationInstance.calibrationFunction(self.data.Amplitude.values), index = self.data.index)
        
        where_tooBig = np.where(self.data.Amplitude > calibrationInstance.data.amp.max())
        where_tooSmall = np.where(self.data.Amplitude < calibrationInstance.data.amp.min())
        too_small = len(where_tooSmall[0])
        too_big = len(where_tooBig[0])
        self.data.Masked.values[where_tooBig] = 2
        self.data.Masked.values[where_tooSmall] = 1
        if verbose:
            misc.msg('\t %s from %s peaks (%.1i %%) are outside the calibration range (amplitude = [%s, %s], diameter = [%s, %s])'%(too_small + too_big, len(self.data.Amplitude),100 * float(too_small + too_big)/float(len(self.data.Amplitude)) , calibrationInstance.data.amp.min(),  calibrationInstance.data.amp.max(), calibrationInstance.data.d.min(), calibrationInstance.data.d.max()))
            misc.msg('\t\t %s too small'%(too_small))
            misc.msg('\t\t %s too big'%(too_big))
        self.particles_larger_than_pops_detection_range = too_big
        return
        
    #########
    ### Plot some stuff
        
    def plot_timeVindex(self):
        notMasked = np.where(self.data.Masked == 0)
        f,a = plt.subplots()
        g, = a.plot(self.data.index[notMasked],'o')
        a.set_title('Time as a function of particle index')
        a.set_ylabel("Time (UTC)")
        a.set_xlabel('Particle index')
        return f,a,g
    
    def _plot_somethingVtime(self, what,title,ylabel):
        notMasked = np.where(self.data.Masked == 0)
        f,a = plt.subplots()
        g, = a.plot(self.data.index[notMasked], self.data[what].values[notMasked],'o')
        a.set_title(title)
        a.set_xlabel("Time (UTC)")
        a.set_ylabel(ylabel)
        return f,a,g
    
    def plot_widthVtime(self):
        return self._plot_somethingVtime('Width','Peak width as a function of time','Width (sampling steps)')
    
    
    def plot_diameterVtime(self):
        return self._plot_somethingVtime('Diameter', 'Peak diameter as a function of time','Diameter (nm)')
    
    
    def plot_amplitudeVtime(self):
        return self._plot_somethingVtime('Amplitude','Peak amplitude as a function of time', 'Amplitude (digitizer bins)' )
        
    ##########
    ##### Analytics
        
    def get_countRate(self,average = None):
        """average: string, e.g. "5S" for 5 seconds
        returns a pandas dataframe"""
        notMasked = np.where(self.data.Masked == 0)
        unique = np.unique(self.data.index.values[notMasked])
        numbers = np.zeros(unique.shape)
        deltaT = np.zeros(unique.shape)
        countsPerSec = np.zeros(unique.shape)
        dt = unique[1]-unique[0]
        for e,i in enumerate(unique):
            if e:
                dt = unique[e] - unique[e-1]
            numbers[e] =  float(len(np.where(self.data.index.values[notMasked] == i)[0]))
            deltaT[e] = dt / np.timedelta64(1, 's') # delta in seconds
            countsPerSec[e] = numbers[e] / deltaT[e]
    
        countRate = pd.DataFrame(np.array([numbers,deltaT,countsPerSec]).transpose(), index = unique, columns=['No_of_particles', 'DeltaT_s', 'CountRate_s'])    
        if average:
            countRate = countRate.resample(average,closed = 'right', label='center')
        return countRate
        

    
    
    def _peak2Distribution(self, bins=defaultBins, distributionType = 'number', differentialStyle = False, fill_data_gaps_with = None, ignore_data_gap_error = False):
        """Action required: clean up!
        Returns the particle size distribution normalized in various ways
        distributionType
        dNdDp, should be fixed to that, change to other types later once the distribution is created!
        old:
            \t calibration: this will create a intensity distribution instead of size distribution. bins should only be a number of bins which will be logaritmically spaced
            \t number:\t numbers only $\mu m^{-1}\, cm^{-3}$
            \t surface:\t surface area distribution, unit: $\mu m\, cm^{-3}$
            \t volume:\t  volume distribution, unit: $\mu m^{2}\, cm^{-3}$
        differentialStyle:\t     if False a raw histogram will be created, else:
            \t dNdDp: \t      distribution normalized to the bin width, bincenters are given by (Dn+Dn+1)/2
            \t dNdlogDp:\t    distribution normalized to the log of the bin width, bincenters are given by 10**((logDn+logDn+1)/2)
    
        """
        notMasked = np.where(self.data.Masked == 0)
        print(notMasked, self.data, self.data.shape)
        # too_big_condi = np.where(self.data.Masked == 2)
        print(self.data.index.values[notMasked])
        unique = np.unique(self.data.index.values[notMasked])
        N = np.zeros((unique.shape[0],bins.shape[0]-1))
        too_big = np.zeros(unique.shape[0])
    
        for e,i in enumerate(unique):
            condi = np.where(np.logical_and(self.data.Masked == 0, self.data.index.values == i))
            if distributionType == 'calibration':
                process = self.data.Amplitude.values[condi]
            else:
                process = self.data.Diameter.values[condi]
            n,edg = np.histogram(process, bins = bins)
            N[e] = n
            too_big[e] = np.logical_and(self.data.Masked == 2, self.data.index.values == i).sum()
    
        N = N.astype(np.float)
        too_big = too_big.astype(np.float)

        print(unique, unique[1:], unique[:-1], np.timedelta64(1, 's'))
        deltaT = (unique[1:]-unique[:-1]) / np.timedelta64(1,'s')
        print(deltaT)
        deltaT_sl = np.append(deltaT[0],deltaT)
        deltaT = np.repeat(np.array([deltaT_sl]),bins.shape[0]-1, axis=0)
        N/=deltaT.transpose()
        too_big /= deltaT_sl.transpose()
        binwidth = edg[1:] - edg[:-1]
    
        if not differentialStyle:
            pass
    
        elif differentialStyle == 'dNdDp':
            N = N/binwidth
        else:
            raise ValueError('wrong type for argument "differentialStyle"')      

        binstr = bins.astype(int).astype(str)
        cols=[]
        for e,i in enumerate(binstr[:-1]):
            cols.append(i+'-'+binstr[e+1])
        dataFrame = pd.DataFrame(N, columns=cols, index = unique)
        # too_big = pd.DataFrame(too_big, columns=['# too big'])
        too_big = _timeseries.TimeSeries(pd.DataFrame(too_big, columns=['# too big'], index = unique))
        if distributionType == 'calibration':
            return sizedistribution.SizeDist_TS(dataFrame, bins, 'calibration')
        else:
            dist = sizedistribution.SizeDist_TS(dataFrame, bins, 'dNdDp',fill_data_gaps_with = fill_data_gaps_with, ignore_data_gap_error = ignore_data_gap_error)
            dist = dist.convert2dNdlogDp()
            dist.particle_number_concentration_outside_range = too_big
            return dist
        
#    def peak2numberdistribution_dNdlogDp(self, bins = defaultBins):
#        return self._peak2Distribution(bins = bins, differentialStyle='dNdlogDp')
#        
#    def peak2numberconcentration(self, bins = defaultBins):
#        return self._peak2Distribution(bins = bins)
    def peak2peakHeightDistribution(self, bins = np.logspace(np.log10(35),np.log10(65000), 200)):
        """see doc-string of _peak2Distribution"""
        return self._peak2Distribution(bins = bins,distributionType = 'calibration',differentialStyle = 'dNdDp')
        
    def peak2sizedistribution(self, bins = 'default', fill_data_gaps_with = None, ignore_data_gap_error = False):
        """see doc-string of _peak2Distribution"""
        if type(bins) == str:
            if bins == 'default':
                bins = defaultBins
        dist = self._peak2Distribution(bins=bins, differentialStyle='dNdDp', fill_data_gaps_with = fill_data_gaps_with, ignore_data_gap_error = ignore_data_gap_error)
        return dist
        
#    def peak2calibration(self, bins = 200, ampMin = 20):
#        bins = np.logspace(np.log10(20), np.log10(self.data.Amplitude.values.max()),bins)
#        dist = self._peak2Distribution(bins = bins, distributionType='calibration')
#        return dist
#    peak2calibration.__doc__ = blabla
