# -*- coding: utf-8 -*-
"""
Contains a cores set of calculations to be performed on particles. 
Atmospheric variables are generally decared with the units of temperature
in degrees Celsius, pressure in millibars and diameters in nanometers.

@author: mtat76
@date: 03-03-2015
"""
#from atmPy 
import constants
from scipy.optimize import fsolve
import math


def Z(D,gas):
    """
    Calculate electric mobility of particle with diameter D. 
    
    Parameters
    -----------
    D: diameter in nm
    gas: object of type gas
    
    Returns
    ---------   
    Electrical mobility in m2/V*s as defined in Hinds (1999), p. 322, eq. 15.21.
    """
      
    try:
        return constants.e*Cc(D,gas)/(3*math.pi*gas.mu()*D*1e-9)
    except AttributeError:
        print('Incorrect type selected for attribute "gas".')
        return 0
    

def Z2D(Z,gas,n):
    '''
    Retrieve particle diameter from the electrical mobility
    
    Call this using a roots or fsolve function.
    
    Parameters
    -----------
    T:  Temperature in degrees Celsius
    P:  Pressure in millibars
    Z:  Electrical mobility in m2/Vs
    n:  Number of charges
    
    Returns
    -------
    Diameter of particle in nanometers.
    '''
    
    # Inline function use with fsolve
    f = lambda d: d*1e-9-n*constants.e*Cc(d,gas)/(3*math.pi*gas.mu()*Z)
    d0 = 1e-9
    return fsolve(f,d0)[0]

    

def Cc(D,gas):
    """
Calculate Cunningham correction factor.

Parameters
-----------
D:      Particle diameter in nanometers.
gas:    gas object from the atmosphere package.

Returns
--------
Cunningham correction factor as a function of diameter and mean free path. 

Notes
-------
This is from Hinds (1999); p49, eq 3.20
    """
    
    # Convert diameter to microns.
    D = float(D)*1e-3
    # Get the mean free path
    try:
        mfp = gas.l()
        return (1.05*math.exp(-0.39*D/mfp)+2.34)*mfp/D+1
        
    except AttributeError:
        print('Invalid type entered for "gas".  Should be of type atmosphere.gas".')
        return 0;
    
def ndistr(dp,n=-1,T=20):
    '''
Bipolar charge distribution.

Parameters
-----------
dp: diameter of particle in nm
n: number of charges
T: temperature in degree C

Returns
--------
Charging efficiency

Notes
------
* For particles smaller than 1 micron, uses Wiedensohler (1988), J. Aerosol Sci., 19, 3.
* For particles larger than 1 micron, uses Gunn (1956), J. Colloid Sci., 11, 661.
    '''
    
    dp = float(dp)
    if (dp<1000):
        if (n==-2):
            a0=-26.3328
            a1=35.9044
            a2=-21.4608
            a3=7.0867
            a4=-1.3088
            a5=0.1051
        elif (n==-1):
            a0=-2.3197
            a1=0.6175
            a2=0.6201
            a3=-0.1105
            a4=-0.1260
            a5=0.0297
        elif (n==0):
            a0 = -0.0003
            a1 = -0.1014
            a2 = 0.3073
            a3 = -0.3372
            a4 = 0.1023
            a5 = -0.0105
        elif (n==1):
            a0 = -2.3484
            a1 = 0.6044
            a2 = 0.4800
            a3 = 0.0013
            a4 = -0.1553
            a5 = 0.0320
        elif (n==2):
            a0 = -44.4756
            a1 = 79.3772
            a2 = -62.8900
            a3 = 26.4492
            a4 = -5.7480
            a5 = 0.5049
        return 10**(a0+a1*math.log10(dp)+a2*math.log10(dp)\
                **2+a3*math.log10(dp)**3+a4*math.log10(dp)**4+a5*math.log10(dp)**5)
    else:
        T+=273.15       #convert [°C] to [K]
        dp*=1e-9         #convert [nm] to [m]
        ionconcrat=1     # ratio of positive and negative ion concentrations
        ionmobrat=0.875  # ratio of positive and negative ion mobilities
        f1=constants.e/math.sqrt(4*math.pi**2*constants.eps0*dp*constants.k*T)
        f2=2*math.pi*constants.eps0*dp*constants.k*T/constants.e**2
        return f1*math.exp(-1*(n-f2*math.log(ionconcrat*ionmobrat))**2/(2*f2))

    
def d50(N,rhop,Q,gas,Dj):
    '''
Find the impactor cutpoint.

Parameters
----------
N: number of jets
rhop: particle density in kg/m^3
Q: volumetric flow rate in lpm
T: temperature in degrees Celsius
P: pressure in millibars
Dj: jet diaemter in meters

Returns
-------
Impactor cutpoint in microns.

Notes
------
Calculations are those of Hinds (1999)
    
    '''
    mu_air = gas.mu()
    #Convert the volumetric flow rate into m^3/s
    Q = float(Q)/60*1000/100**3

    # From Hinds, this is the Stoke's number for the 50% collections
    # efficiency (Table 5.4)
    Stk50 = 0.24

    # Equation 5.29 in Hinds (1999)
    D50Cc = math.sqrt(9*mu_air*math.pi*N*Dj**3*Stk50/(4.0*float(rhop)*Q))

    f = lambda x: (D50Cc/float(x))**2-Cc(float(x*1e-6),gas)
    
    # Find the D50 of the impactor
    return fsolve(f, 0.1)
    
    
    
