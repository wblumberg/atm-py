# -*- coding: utf-8 -*-
"""
This package contains routines required for performing 
common atmospheric calculations that might be required
for aerosol analysis.

@author: mtat76
"""

from math import *
from constants import *

def mu(T):    
    """
    The following function defines viscocity as a function of T in P-s.
    @param T:   temperature in Kelvin
    @return:    Viscocity in P-s
    """
    
    # Make sure that the temperature is a float
    T = float(T)
    C = 120.0 # Sutherland's constant
    mu0 = 18.27e-6 # Reference viscocity
    T0 = 291.15 # Reference temperature
    
    return (C+T0)/(C+T)*pow((T/T0),1.5)*mu0
    

def l(P):    
    """
    Determine the mean free path of air.
    @param P:   pressure in millibars.
    @return:    Mean free path of air in microns.
    """
    # Convert pressure to atmospheres
    P = float(P)/1013.25
    l0 = 0.066 # Reference mean free path at 1 atm
    
    # Return mean free path in microns
    return l0/P
    