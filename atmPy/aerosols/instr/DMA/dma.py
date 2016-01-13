# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:45:47 2015

@author: mrichardson
"""
from math import log
from math import pi

from scipy.optimize import newton

from atmPy.aerosols.aerosol import z


class DMA(object):
    """
    Class describing attributes and methods unique to the differential mobility analyzer.

    This class may be instantiated directly or utilize one of the prebuilt child classes.

    Attributes
    ----------
    ro: float
        Outer radius of DMA in meters.
    ri: float
        Inner radius of DMA in meters.
    l:  float
        Length of DMA column in meters
    """
    def __init__(self, l, ro, ri):
        self._ro = ro
        self._ri = ri
        self._l = l

    def v2d(self, v, gas, qc, qm=None):
        """
        Find selected diameter at a given voltage.
        
        This function uses a Newton-Raphson root finder to solve for the 
        diameter of the particle.
        
        Parameters
        ----------
        v:      float
                Voltage in Volts
        gas:    gas object
                Carrier gas object for performing calculations
        qc:     float
                Input sheath flow in lpm
        qm:     float, optional
                Output sheath flow in lpm; default is the input sheath flow
        
        Returns
        -------
        Diameter in nanometers
    
        """
        gamma = self._l/log(self._ro/self._ri)

        # Convert flow rates from lpm to m3/s
        qc = float(qc)/60*0.001

        if qm is None:
            qm = qc
        else:
            qm = float(qm)/60*0.001
        
        # Central mobility
        zc = (qc+qm)/(4*pi*gamma*v)
        return newton(lambda d: z(d, gas, 1)-zc, 1, maxiter=1000)

    def xfer(self, qa, qc,  gas, v, qm=None, qs=None):
        """
        Calculates the transfer function Omega described in Knutson and Whitby, 1975 (equation 12).

        Parameters
        -----------
        qa: float
            aerosol flow rate in lpm
        qc: float
            clean air flow rate in lpm
        gas: Gas object
            Instance of the carrier gas object that contains the current conditions
        V: float
            DMA center rod voltage in V
        qs: float, optional
            sampling flow rate in lpm, set to qa if not entered
        qm: float, optional
            main outlet flow rate in lpm, set to qc if not entered

        Returns
        -------
        The probability that an aerosol particle which enters the DMA will leave via
        the sample flow given the mobility Z
        """

        # Convert flow to m3/s
        qa = float(qa)/60*0.001
        qc = float(qc)/60*0.001

        if qm is None:
            qm = qc
        else:
            qm = float(qm)/60*0.001

        if qs is None:
            qs = qa
        else:
            qs = float(qs)/60*0.001

        # lambda from Knutson and Whitby equation (14)
        l = self._l/log(self._ri/self._ro)

        # Difference between aerosol in and aerosol out electric flux function
        dphi = l*v

        # Mobility centroid from Knutson and Whitby, equation (15)
        zp = (qa+qs)/(4*pi*dphi)

        # return the value of omega; this is equation (12) Knutson and Whitby, 1975
        return 1/qa*max(0, min(qa, qs, 0.5(qa+qs)-abs(2*pi*zp*dphi+0.5*(qm+qc))))


class NoaaWide(DMA):
    """
    Sets the dimensions to those of the DMA developed by NOAA.
    """

    def __init__(self):
        super(NoaaWide, self).__init__(0.34054, 0.03613, 0.0312)


class Tsi3071(DMA):
    """
    Child of DMA which contains the dimensions for the TSI 3071 DMA.
    """
    def __init__(self):
        super(Tsi3071,self).__init__(0.4444, 0.0195834, 0.0093726)


class Tsi3081(DMA):
    """
    Child of DMA which contains the dimensions for the TSI 3081 DMA.
    """
    def __init__(self):
        super(Tsi3081,self).__init__(0.44369, 0.01961, 0.00937)


class Tsi3085(DMA):
    """
    Defines the dimensions of a DMA with the TSI nano DMA dimensions.
    """

    def __init__(self):
        super(Tsi3085, self).__init__(0.04987, 0.01961, 0.00937)
