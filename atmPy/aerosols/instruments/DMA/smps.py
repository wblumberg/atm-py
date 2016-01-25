import sys
import tkinter as tk
from datetime import datetime as dt
from datetime import timedelta
from math import floor
from tkinter import filedialog as fd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import interp1d

from atmPy.general.atmosphere import Air
from atmPy.aerosols.physics import aerosol


class SMPS(object):
    """
    Defines routines associated with an SMPS object

    Attributes
    ----------
    dma:            DMA object
                    Defines the differential mobility analyzer specific to the SMPS
    dn_interp:      numpy array of floats, 2D
                    Charge and lag corrected dndlogDp distributions
    diam_interp:    numpy array of floats, 1D
                    Diameters over which the dndlogDp distributions are interpolated.  This value is determined by the
                    method buildGrid
    cn_raw:
    cn_smoothed:
    diam:
    air:            Air object
                    Use this object to set the temperature and pressure and do calculations related to the gas
    files:          list of file objects
                    These are the files that will be processed
    scan_folder:    String
                    Location of scan data


    """
    def __init__(self, dma):

        self.air = Air()
        self.dma = dma
        self.files = []

        self.diam_interp = 0

        self.buildGrid()

        self.dn_interp = None
        self.date = []
        self.cn_raw = None
        self.cn_smoothed = None
        self.diam = None
        self.lag = 10
        # Smoothing parameter for the LOWESS smoothing
        self.alpha = 0.3

    def openFiles(self, scan_folder=''):
        """
        Launches a dialog to selct files for SMPS processing.

        Parameters
        -----------
        scan_folder:    string, optional
                        Starting folder for searching for files to process.  Default is empty.
        """
        _gui = tk.Tk()

        # Prompt the user for a file or files to process
        self.files = fd.askopenfiles(initialdi=scan_folder)
        _gui.destroy()

    def __progress__(self):
        s = len(self.files)
        i = 0
        while True:
            print('\r[{0}] {1}%'.format('#'*i, i/s*100))
            yield None
            i += 1

    @staticmethod
    def __chargecorr__(diam, dn, gas, n=3, pos_neg=-1):
        """
        Correct the input concentrations for multiple charges.

        When running in SMPS mode, we must keep in mind that each size of particles can carry multiple charges.
        Particles with more charges will appear to be SMALLER than those with less.  So, we will index through the
        diameters backwards, get the number of the current selected size that would have charge n, and remove those
        from the lower bins to the upper bins.

        Parameters
        ----------
        diam:       array of float
                    array of diameters in nm
        dn:         array of integers
                    Array of particle concentrations corresponding to diameter 'diam'
        gas:        gas object
                    Gas object that defines the properties of the gas
        n:          int, optional
                    Number of charges to consider.  Default is 3.
        pos_neg:    int, optional
                    Positive or negative one indicating whether to consider positive or negative charges.
                    Default is -1.

        Returns
        -------
        None

        Notes
        ------
        The charge correction loops from top to bottom in the size distribution.
        This assumes that there are no particles beyond the probed distribution.
        If there are, there will be noticeable steps in the charge corrected distribution.

        Regarding the algorithm, we can solve for each bin by using the charging efficiency
        ala Gunn or Wiedensohler.  To solve, we use the following algorithm:

        1. Assume that the current bin holds ONLY singly charged particles.

        2. Calculate the fraction of particles that are expected to be singly charged
        at the current diameter, :math:'f_1\left(D_p\right)'.

        3. For each charge beyond 1:

            a) Get the charging efficiency of the current particle size for :math:'i' charges
            :math:'f_i \left(D_p\right)'

            b)

        """
        # Flip the incoming diamter array
        rdiam = np.copy(diam[::-1])

        # We are working backwards, so we need to have the length to get this all right...
        l = len(dn)-1

        # Inline function for finding the value closest to diameter d in the array diam
        fmin = lambda nd: (np.abs(np.asarray(diam) - nd)).argmin()

        for i, d in enumerate(rdiam):

            # Get the fraction of particles that are singly charged
            f1 = aerosol.ndistr(d, pos_neg, gas.t)

            for j in reversed(range(2, n+1)):  # Loop through charges 2 and higher backwards

                ne = j*pos_neg
                fi = aerosol.ndistr(d, ne, gas.t)

                # Mobility of multiply charged particles
                z_mult = abs(ne * aerosol.z(d, gas, pos_neg))

                # Diameter bin which contains the multiply charged particles
                d_mult = aerosol.z2d(z_mult, gas, 1)

                # Continue while the diameter specified is larger than the minimum diameter
                if d_mult >= diam[0]:

                    # Find the index of the multiple charges
                    k = fmin(d_mult)

                    # Remove the particles in bin k that belong in the current bin, but don't remove more
                    # particles than there are in the bin
                    dn[k] -= min(dn[l-i]*fi/f1, dn[k])

                # The total number of particlesi n the current bin is simply the number of singly charged
                # particles divided by the singly charged charging efficiency.
            dn[l-i] /= f1

        return None

    def proc_files(self):
        """
        Process the files that are contained by the SMPS class attribute 'files'
        """

        e_count = 0
        # TODO: Some of the processes here can be parallelized using the multiprocessing library
        #       Reading of the data and determing the lag will require a proper sequence, but
        #       we can parallelize:
        #       1) The processing of an individual file
        #       2) The processing of the up and down scans within a file
        #       WARNING:    Need to ensure that the different processes don't step on each other.
        #                   Likely we would need to make some of the instance variables local (air.t and air.p
        #                   come to mind).

        self.dn_interp = np.zeros((2*len(self.files), len(self.diam_interp)))
        self.date = [None]*2*len(self.files)

        self.cn_raw = np.zeros((2*len(self.files), len(self.diam_interp)))
        self.cn_smoothed = np.zeros((2*len(self.files), len(self.diam_interp)))
        self.diam = np.zeros((2*len(self.files), len(self.diam_interp)))

        for e, i in enumerate(self.files):

            try:

                print(i.name)

                # Get the data in the file header
                meta_data = self.__readmeta__(i.name)

                # Retrieve the scan and dwell times from the meta data.
                tscan = meta_data['Scan_Time'].values[0]
                tdwell = meta_data['Dwell_Time'].values[0]

                # This is the data in the scan file
                data = self.__readdata__(i.name)

                # Get the CPC data of interest and pad the end with zeros for the sake of
                # readability.
                cpc_data = np.pad(data.CPC_1_Cnt.values[self.lag:], (0, self.lag),
                                  mode="constant", constant_values=(0, 0))

                # This is the CPC concentration
                cpc_data /= data.CPC_Flw.values

                # Remove NaNs and infs from the cpc data
                cpc_data[np.where(np.isnan(cpc_data))] = 0.0
                cpc_data[np.where(np.isinf(cpc_data))] = 0.0

                # In the following section, we will take the two variables, 'data' and
                # 'cpc_data' to produce the data that will be run through the core of
                # the processing code.  The steps are as follows to prepare the data:
                #   1. If the data is the downward data, flip the arrays.
                #   2. Truncate the data to get the scanned data.
                #       a. If the data is the up data, we simply want the first 'tscan'
                #          elements.
                #       b. If the data is the down data, we will account for the final
                #          dwell time ('tdwell'), and take the portion of the arrays
                #          from tdwell to tscan + tdwell.
                #   3. Get the mean values of all the data in the scan array from the
                #      respective data array.  We will use the mean values for inversion.

                # PRODUCE UP DATA FOR PROCESSING #
                # Extract the portion of the CPC data of interest for the upward scan
                cpc_up = cpc_data[:tscan]
                up_data = data.iloc[:tscan]
                smooth_up = sm.nonparametric.lowess(cpc_up, up_data.DMA_Diam.values,
                                                    frac=self.alpha, it=1, missing='none',
                                                    return_sorted=False)

                smooth_up[np.where(np.isnan(smooth_up))] = 0.0
                smooth_up[np.where(np.isinf(smooth_up))] = 0.0

                # Retrieve mean up data
                mup = up_data.mean(axis=0)

                self.air.t = mup.Aer_Temp_C
                self.air.p = mup.Aer_Pres_PSI

                # Calculate diameters from voltages
                dup = [self.dma.v2d(i, self.air, mup.Sh_Q_VLPM,
                                    mup.Sh_Q_VLPM) for i in up_data.DMA_Set_Volts.values]

                # UP DATA PRODUCTION COMPLETE #

                # BEGIN DOWN DATA PRODUCTION #
                # Flip the cpc data and extricate the portion of interest
                cpc_down = cpc_data[::-1]
                cpc_down = cpc_down[tdwell:tscan+tdwell]

                # Flip the down data and slice it
                down_data = data.iloc[::-1]
                down_data = down_data.iloc[int(tdwell):int(tscan+tdwell)]

                smooth_down = sm.nonparametric.lowess(cpc_down, down_data.DMA_Diam.values,
                                                      frac=self.alpha, it=1, missing='none',
                                                      return_sorted=False)

                smooth_down[np.where(np.isnan(smooth_up))] = 0.0
                smooth_down[np.where(np.isinf(smooth_up))] = 0.0

                # Retrieve mean down data
                mdown = down_data.mean(axis=0)

                self.air.t = mdown.Aer_Temp_C
                self.air.p = mdown.Aer_Pres_PSI

                # Calculate diameters from voltages
                ddown = [self.dma.v2d(i, self.air, mdown.Sh_Q_VLPM,
                                      mdown.Sh_Q_VLPM) for i in down_data.DMA_Set_Volts.values]

                up_interp_dn = self.__fwhm__(dup, smooth_up, mup)
                down_interp_dn = self.__fwhm__(ddown, smooth_down, mdown)

                up_interp_dn[np.where(up_interp_dn < 0)] = 0
                down_interp_dn[np.where(down_interp_dn < 0)] = 0

            except ValueError:
                print("Unexpected error:", sys.exc_info()[0])
                print("Issue processing file " + str(i.name))
                e_count += 1
                continue
            except TypeError:
                print("Error processing file.")
                e_count += 1
                continue

            else:
                n_e = e-e_count
                # Stuff the data for the attributes down here.  If an error is thrown, we will not contaminate
                # member data.

                self.date[2*n_e] = dt.strptime(str(meta_data.Date[0]) + ',' + str(meta_data.Time[0]),
                                               '%m/%d/%y,%H:%M:%S')
                self.date[2*n_e + 1] = self.date[2*n_e] + timedelta(0, tscan + tdwell)

                self.diam[2*n_e, 0:np.asarray(dup).size] = np.asarray(dup)
                self.cn_raw[2*n_e, 0:cpc_up.size] = cpc_up

                # Store raw data for the down scan
                self.cn_raw[2*n_e+1, 0:cpc_down.size] = cpc_down
                self.diam[2*n_e+1, 0:np.asarray(ddown).size] = np.asarray(ddown)

                self.cn_smoothed[2*n_e, 0:smooth_up.size] = smooth_up
                self.cn_smoothed[2*n_e+1, 0:smooth_down.size] = smooth_down
                self.dn_interp[2*n_e, :] = up_interp_dn
                self.dn_interp[2*n_e+1, :] = down_interp_dn

    @staticmethod
    def __readmeta__(file):

        """
        Parameters
        ----------
        file:   file object
                File to retrieve meta data from

        Returns
        -------
        pandas datafram containing meta data
        """
        return pd.read_csv(file, header=0, lineterminator='\n', nrows=1)

    @staticmethod
    def __readdata__(file):
        """
        Read the data from the file.

        Data starts in the third row.

        Parameters
        -----------
        file:   file object
                File containing SMPS data

        Returns
        --------
        pandas data frame
        """
        return pd.read_csv(file, parse_dates='Date_Time', index_col=0, header=2, lineterminator='\n')

    def getLag(self, index, delta=0, p=True):
        """
        This function can be called to guide the user in how to set the lag attribute.

        Parameters
        ----------
        index:  int
                Index of file in attribute files to determine the lag
        delta:  int, optional
                Fudge factor for aligning the two scans; default is 0
        p:      Boolean, optional
                Plot the output if True
        """
        meta_data = self.__readmeta__(self.files[index].name)
        up_data = self.__readdata__(self.files[index].name)

        tscan = meta_data.Scan_Time.values[0]
        tdwell = meta_data.Dwell_Time.values[0]

        # CPC concentrations will be simply the 1 s buffer divided by the
        # CPC flow
        cpc_cnt = up_data.CPC_1_Cnt.values/up_data.CPC_Flw.values

        # Truncate the upward trajectory
        up = cpc_cnt[0:tscan]

        # Get the counts for the decreasing voltage and truncate them
        down = cpc_cnt[::-1]                    # Flip the variable cpc_cnt
        down = cpc_cnt[tdwell:(tscan+tdwell)]     # Truncate the data
        down_data = up_data[::-1]

        # LAG CORRELATION #
        corr = np.correlate(up, down, mode="full")
        plt.plot(corr)
        corr = corr[corr.size/2:]
        self.lag = floor(corr.argmax(axis=0)/2+delta)

        f = self.lag

        # GET CPC DATA FOR PLOTTING #
        # Shift the up data with the number of zeros padding on the end equal to the lag
        up = up_data['CPC_1_Cnt'].values[f:tscan+f]/up_data['CPC_Flw'].values[f:tscan+f]

        # Remove NaNs and infs
        up[np.where(np.isinf(up))] = 0.0
        up[np.where(np.isnan(up))] = 0.0

        up_data = up_data.iloc[:tscan]
        smooth_p = 0.3
        smooth_up = sm.nonparametric.lowess(up, up_data.DMA_Diam.values, frac=smooth_p, it=1, missing='none')

        # Padding the down scan is trickier - if the parameter f (should be the lag in the correlation)
        # is larger than the dwell time, we will have a negative resize parameter - this is no good.
        # Pad the front with the number of zeros that goes beyond the end (front in the reveresed array).
        # This makes sense.  I guess.

        if f > tdwell:
            f = tdwell
            down = np.pad(down_data['CPC_1_Cnt'].values[0:tscan] /
                          down_data['CPC_Flw'].values[0:tscan],
                          pad_width={tdwell-f, 0}, constant_values={0, 0})
        else:
            down = (down_data['CPC_1_Cnt'].values[(tdwell-f):(tscan+tdwell-f)] /
                    down_data['CPC_Flw'].values[(tdwell-f):(tscan+tdwell-f)])

        down[np.where(np.isinf(down))] = 0.0
        down[np.where(np.isnan(down))] = 0.0
        down_data = down_data.iloc[:tscan]
        smooth_down = sm.nonparametric.lowess(down, down_data.DMA_Diam.values, frac=smooth_p, missing='none')

        if p:
            f2 = plt.figure(2)

            plt.plot(up, 'r.', down, 'b.', smooth_up[:, 1], 'r+', smooth_down[:, 1], 'b+')
            output = "Lag is estimated to be " + str(self.lag) + " with a delta of " + str(delta) + "."
            plt.title(output)
            plt.show()

    def buildGrid(self, logtype="ln", gmin=1, gmax=1000, n=300):
        """
        Define a logarithmic grid over which to interpolate output values

        Parameters
        ----------
        type:   string, optional
                this value can be log10 or natural (e); default is log10
        min:    int, optional
                minimum value in the grid; default is 1
        max:    int, optional
                maximum value in the grid; default is 1000
        n:      int, optional
                number of bins over which to divide the grid; default is 300
        """
        if logtype == "log10":
            self.diam_interp = np.logspace(np.log10(gmin), np.log10(gmax), n, endpoint=True)
        else:
            self.diam_interp = np.logspace(np.log(gmin), np.log(gmax), n, base=np.exp(1), endpoint=True)

        return None

    def __fwhm__(self, diam, dn, mean_data):
        """
        Retrieve the full width at half max and return an interpolated concentration dN/dlogdp array

        Parameters
        -----------
        diam:       NumPy array of floats
                    Diameters calculated from the setpoint voltage of the scan.  Units are nm
        dn:         NumPy array of floats
                    CPC concentration at each diameter.  Units are cc^-1.
        mean_data:  pandas DataFrame
                    DataFrame containing mean data from the scan.
        :return:
        """
        ls = len(dn)
        dlogd = np.zeros(ls)    # calculate dlogd
        fwhm = np.zeros(ls)     # hold width
        self.air.t = mean_data.Aer_Temp_C
        self.air.p = mean_data.Aer_Pres_PSI

        def xfer(dp, qa, qs):
            """
            Return the full-width, half-max of the transfer function in diameter space.
            This implementation ignores diffusion broadening.

            Parameters
            -----------
            dp: float
                particle size in nm
            qa: float
                aerosol flow rate in lpm
            qs: float
                aerosol flow rate in lpm

            Returns
            -------
            Width of transfer function in nm.
            """
            beta = float(qa)/float(qs)

            # Retrieve the center mobility
            zc = aerosol.z(dp, self.air, 1)

            # Upper bound of the mobility
            zm = (1-beta/2)*zc

            # Lower bound of the mobility
            zp = (1+beta/2)*zc

            return aerosol.z2d(zm, self.air, 1) - aerosol.z2d(zp, self.air, 1)

        for e, i in enumerate(diam):
            try:
                fwhm[e] = xfer(i, mean_data.Aer_Q_VLPM, mean_data.Sh_Q_VLPM)
                dlogd[e] = np.log10(i+fwhm[e]/2)-np.log10(i-fwhm[e]/2)

            except (ValueError, ZeroDivisionError):
                fwhm[e] = np.nan
                print('Handling divide by zero error')
            except:
                fwhm[e] = np.nan
                print('Handling unknown error: ' + str(sys.exc_info()[0]))

        # Correct for multiple charging.  We will use the array dn by reference and stuff this
        # into another array
        self.__chargecorr__(diam, dn, self.air)

        output_sd = np.copy(dn)

        # Divide the concentration by dlogdp from the transfer function
        output_sd /= dlogd

        # Use the 1D interpolation scheme to project the current concentrations
        # onto the array defined by diam_interp
        f = interp1d(diam, output_sd, bounds_error=False, kind='linear')

        # Return the interpolated dNdlogDp distribution
        return f(self.diam_interp)




