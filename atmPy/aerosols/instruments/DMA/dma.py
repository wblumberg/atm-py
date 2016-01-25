# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:45:47 2015

@author: mrichardson
"""
from math import log
from math import pi

from scipy.optimize import newton

from atmPy.aerosols.physics.aerosol import z


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

    def omega(self, v, qa, qs):
        """
        Calculate the DMA transfer function.

        This is the function as presented in Knutson and Whitby [1975] on the differential mobility analyzer.

        This does not account for instances where Qa ~= Qm or Qs ~= Qe.

        Parameters
        -----------
        v:
        qa:
        qs:

        """
        qa = float(qa)/60*0.001
        qs = float(qs)/60*0.001
        l = self._l/log(self._ro/self.ri)

        # Electric flux function band
        dphi = self.l*v/l

        # Mobility centroid
        zp = (qa+qs)/(4*pi*l*v)

        return 1/qa*max(0, min([qa, qs, qa-abs(2*pi*zp*dphi+qs)]))

    def v2d(self, v, gas, qc, qm):
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
        qm:     float
                Output sheath flow in lpm
        
        Returns
        -------
        Diameter in nanometers
    
        """
        gamma = self._l/log(self._ro/self._ri)

        # Convert flow rates from lpm to m3/s
        qc = float(qc)/60*0.001
        qm = float(qm)/60*0.001
        
        # Central mobility
        zc = (qc+qm)/(4*pi*gamma*v)
        return newton(lambda d: z(d, gas, 1)-zc, 1, maxiter=1000)


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
