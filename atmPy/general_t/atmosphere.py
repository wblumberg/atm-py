# -*- coding: utf-8 -*-
"""
This package contains routines required for performing 
common atmospheric calculations that might be required
for aerosol analysis.

@author: mtat76
"""

from math import log10
from math import tanh
from math import log
from math import exp
import abc

class Esat:
    """
    This class uses a strategy pattern to determine how saturation vapor
    pressures will be calculated.

    Notes
    -----
    * This class provides two abstract methods - ei and ew - which calculate the
    water vapor pressure with respect to ice and water (respectively).
    * One of the three available subclasses MUST be used in order to use this class.

    Examples
    --------
    >>>import atmosphere as atmos

    >>>eg = atmos.goff()

    >>>t = 20 # Set the temperature to 20 degrees Celsius

    Print the saturation vapor pressure with respect to water

    >>>print eg.ew(t)

    """
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod   
    def ew(self, t):
        """
        Calculate saturation vapor pressure as a function of temperature.
        
        Parameters
        ----------
        t: temperature in degress Celsius
        
        Returns
        -------
        Saturation vapor pressure with respect to water in millibars.
        """
        return 0
        
    @abc.abstractmethod
    def ei(self, t):
        """
        Calculate saturation vapor pressure with respect to ice.
        
        Parameters
        ---------
        t: temperature in degrees Celsius.
        
        Returns
        -------
        Saturation vapor pressure with respect to water in millibars.
        """
        return 0


class GoffGratch(Esat):
    """
    Use Goff-Gratch equations to calculate vapor pressure over water or ice.
    """
    
    def ew(self, t):
        c = [-7.90298, 5.02808, -1.3816E-07, 11.344, 8.1328E-03, -3.19149]
        ts = 373.16
        t += 273.15
        b = c[3]*(1.0 - t/ts)
        d = c[5]*(ts/t - 1.0)
        a = c[0]*(ts/t - 1.0) + c[1]*log10(ts/t) + \
            c[2]*(10.0**b-1) + c[4]*(10.0**d-1) + log10(1013.246)
        return 10.0**a
         
    def ei(self, t):
        c = [log10(6.1071), -9.09718, -3.56654, 0.876793]
        t += 273.15
        a = c[0] + c[1]*(273.15/t - 1) + c[2]*log10(273.15/t) + c[3]*(1 - t/273.15)
        return 10.0**a


class Buck81(Esat):
    """
    Calculate water vapor pressure as function of temperature according to Buck (1981).
    """
    
    def ew(self, t):
        a = 6.1121
        b = 18.729
        c = 257.87
        d = 227.3
        # This is equation 4a in Buck (1981)
        return a*exp((b-t/d)*t/(t+c))
        
    def ei(self, t):
        # These coefficients are from Table 2
        a = 6.1115
        b = 23.036
        c = 279.82
        d = 333.7
        return a*exp((b - t/d)*t/(t + c))   # This is equation 4a
        
        
class MurphyKoop(Esat):
    
    def ew(self, t):
        t += 273.15
        es = 54.842763 - 6763.22/t - 4.210*log(t) + 0.000367*t +\
            tanh(0.0415*(t - 218.8))*(53.878 - 1331.22/t - 9.44523*log(t) + 0.014025*t)
         
        # Convert from Pa -> hPa (mb)
        return exp(es)/100

    def ei(self, t):
        t += 273.15
        return exp(9.550426 - 5723.265/t + 3.53068*log(t) - 0.00728332*t)/100


class Gas(object):
    """
    Generic object defining different properties of gasses.

    Attributes
    ----------
    p:  float
        Pressure in mb.
    t:  float
        Temperature in degrees Celsius.
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, t=20, p=1013.25):
        self.t = t
        self.p = p

    def __str__(self):
        return "Gas object with T = " + str(self.t) + " and P = " + str(self.p) + "."
    
    @abc.abstractmethod
    def mu(self): 
        return 0
    
    @abc.abstractmethod
    def l(self):
        return 0


class Air(Gas):
    def __init__(self, t=20.0, p=1013.25):
        super(Air, self).__init__(t, p)
        
    def mu(self):    
        """
        The following function defines viscosity as a function of T in P-s.
        
        Parameters
        ---------
        T:temperature in degrees Celsius
        
        Returns
        -------
        Viscosity in P-s
        """

        # Make sure that the temperature is a float
        t = self.t + 273.15
        c = 120.0       # Sutherland's constant
        mu0 = 18.27e-6  # Reference viscocity
        t0 = 291.15     # Reference temperature
        
        return (c+t0)/(c+t)*(t/t0)**1.5*mu0
        
    def l(self):  
        """
        Determine the mean free path of air.
            
        Returns
        -------
        Mean free path of air in microns.
        """

        # Convert pressure to atmospheres
        patm = float(self.p)/1013.25
        l0 = 0.066  # Reference mean free path at 1 atm
        
        return l0/patm