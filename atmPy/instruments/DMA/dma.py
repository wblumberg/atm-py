# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:45:47 2015

@author: mrichardson
"""
import math
from atmPy import aerosol
import scipy.optimize as opt
#import atmosphere as atmos

class dma(object):
    def __init__(self, l, ro,ri):
        self._ro = ro
        self._ri = ri
        self._l = l

    def V2D(self,V, gas,qc,qm):
        '''
        Find selected diameter at a given voltage.
        
        This function uses a Newton-Raphson root finder to solve for the 
        diameter of the particle.
        
        Parameters
        ----------
        V:   Voltage in Volts
        gas: Carrier gas object for performing calculations
        qc:  Input sheath flow in lpm
        qm:  Output sheath flow in lpm
        
        Returns
        -------
        Diameter in nanometers
    
        '''
        gamma = self._l/math.log(self._ro/self._ri)
        #Convert flow rates from lpm to m3/s
        qc = float(qc)/60*0.001
        qm = float(qm)/60*0.001
        
        # Central mobility
        zc = (qc+qm)/(4*math.pi*gamma*V)
        return opt.newton(lambda d:aerosol.Z(d,gas)-zc, 1, maxiter = 1000)
        
class noaa_wide(dma):
    '''
    Sets the dimensions to those of the DMA developed by NOAA.
    '''
    def __init__(self):
        super(noaa_wide,self).__init__(0.34054,0.03613, 0.0312)
        
class tsi_3071(dma):
    def __init__(self):
        super(tsi_3071,self).__init__(0.4444, 0.0195834, 0.0093726)
        
class tsi_3081(dma):
    def __init__(self):
        super(tsi_3081,self).__init__(0.44369, 0.01961, 0.00937)
        
class tsi_3085(dma):
    '''
    Defines the dimensions of a DMA with the TSI nano DMA dimensions.
    '''
    def __init__(self):
        super(tsi_3085,self).__init__(0.04987, 0.01961, 0.00937)
