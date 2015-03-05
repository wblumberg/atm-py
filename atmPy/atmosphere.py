# -*- coding: utf-8 -*-
"""
This package contains routines required for performing 
common atmospheric calculations that might be required
for aerosol analysis.

@author: mtat76
"""

import math
import abc
 

class esat:
    '''
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

>>>T = 20 # Set the temperature to 20 degrees Celsius

Print the saturation vapor pressure with respect to water

>>>print eg.ew(T)

    '''    
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod   
    def ew(self, T):
        '''
        Calculate saturation vapor pressure as a function of temperature.
        
        Parameters
        ----------
        T: temperature in degress Celsius
        
        Returns
        -------
        Saturation vapor pressure with respect to water in millibars.
        '''
        return 0
        
    @abc.abstractmethod
    def ei(self, T):
        '''
        Calcualte saturation vapor pressure with respect to ice.
        
        Parameters
        ---------
        T: temperature in degrees Celsius.
        
        Returns
        -------
        Saturation vapor pressure with respect to water in millibars.
        '''
        return 0
        

        
class goff(esat):
    '''
    
    '''
    
    def ew(self, T):
         C = [ -7.90298, 5.02808, -1.3816E-07, 11.344, 8.1328E-03, -3.19149 ]
         TS = 373.16
         T += 273.15
         B = C[3]*(1.0 - T/TS)
         D = C[5]*(TS/T - 1.0)
         A = C[0]*(TS/T - 1.0) + C[1]*math.log10(TS/T)  + \
             C[2]*(10.0**B-1)+ C[4]*(10.0**D-1) + math.log10(1013.246) 
         return 10.0**A
         
    def ei(self, T):
        c = [math.log10(6.1071), -9.09718, -3.56654, 0.876793]
        T += 273.15
        A = c[0] + c[1]*(273.15/T - 1) + c[2]*math.log10(273.15/T) + c[3]*(1 - T/273.15)
        return 10.0**A
        
class buck81(esat):
    '''
    Calculate water vapor pressure as function of temperature according to Buck (1981).
    '''
    
    def ew(self,T):
        a = 6.1121
        b = 18.729
        c = 257.87
        d = 227.3
        # This is equation 4a in Buck (1981)
        return a*math.exp((b-T/d)*T/(T+c))
        
    def ei(self,T):
        # These coefficients are from Table 2
        a = 6.1115
        b = 23.036
        c = 279.82
        d = 333.7
        return a*math.exp((b-T/d)*T/(T+c)) # This is equation 4a
        
        
class murphkoop(esat):
    
    def ew(self, T):
         T += 273.15
         es = 54.842763 - 6763.22/T - 4.210*math.log(T) + 0.000367*T +\
             math.tanh(0.0415*(T - 218.8))*(53.878 - 1331.22/T - 9.44523*math.log(T) + 0.014025*T);
         
         # Convert from Pa -> hPa (mb)
         return math.exp(es)/100;
         
    def ei(self, T):
        T += 273.15
        return math.exp(9.550426 - 5723.265/T + 3.53068*math.log(T) - 0.00728332*T )/100
        
class gas(object):
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,T=20,P=1013.25):
        self.T = T
        self.P = P
    
    @abc.abstractmethod
    def mu(self): 
        return 0
    
    @abc.abstractmethod
    def l(self):
        return 0
        
class air(gas):
    def __init__(self,T=20,P=1013.25):
        super(air, self).__init__(T,P)
        
    def mu(self):    
        """
        The following function defines viscocity as a function of T in P-s.
        
        Parameters
        ---------
        T:temperature in degrees Celsius
        
        Returns
        -------
        Viscocity in P-s
        """
    
        
        # Make sure that the temperature is a float
        T = float(self.T)+273.15
        C = 120.0 # Sutherland's constant
        mu0 = 18.27e-6 # Reference viscocity
        T0 = 291.15 # Reference temperature
        
        return (C+T0)/(C+T)*(T/T0)**1.5*mu0
        
    def l(self):  
        '''
        Determine the mean free path of air.
    
        Parameters
        ---------
        P:   pressure in millibars.
            
        Returns
        -------
        Mean free path of air in microns.
        '''
        # Convert pressure to atmospheres
        Patm = float(self.P)/1013.25
        l0 = 0.066 # Reference mean free path at 1 atm
        
        return l0/Patm
        
    
        
    
        

        