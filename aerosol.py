# -*- coding: utf-8 -*-
"""
Contains a cores set of calculations to be performed on particles. 
Atmospheric variables are generally decared with the units of temperature
in degrees Celsius, pressure in millibars and diameters in nanometers.

@author: mtat76
"""

from atmosphere import *
from constants import *


def Z(D,T,P):
    """
    @param D: diameter in nm
    @param P: Pressure in mb.
    @param T: Temperature in degrees Celsius.
    @return: Electrical mobility in m2/V*s as defined in Hinds (1999), p. 322, eq. 15.21.
    """
    
    # Convert diameter to meters
    D = float(D)*1e-9
    
    # Convert temperature to Kelvin
    T = float(T)+273.15
    
    return e*Cc(D,P)/(3*pi*mu(T)*D)
    

def Z2D(d, *data):
    '''
    Zero function to retrieve diameter from mobility.
    Call this using a roots or fsolve function.
    @param d:       particle diameter in meters
    @param *data:   tuple containing temperature in degC, pressure in mb, 
                    electrical mobility in m2/(Vs), and number of charges
    @return:        0 at solution
    '''
    T,P,Z,n = data
    T+=273.15 # Convert T to Kelvin
    return d - n*e*Cc(d,P)/(3*pi*mu(T)*Z)

    
"""
This is from Hinds (1999); p49, eq 3.20
@param D: diameter in meters.
@param P: pressure in millibars.
@return: Cunningham correction factor as a function of diameter and mean free path. 
"""
def Cc(D,P):
    
    # Convert diameter to microns.
    D = float(D)*1e6
    # Get the mean free path
    mfp = l(P)
    
    return (1.05*exp(-0.39*D/mfp)+2.34)*mfp/D+1
    

def eq(D, atm, qc,qm, DMA, V):
    """
    Zero function for determining the diameter as a function of voltage.
    Call eq with a zero finding function to determine the diameter at 
    the given conditions
    @param D:   Diameter in nm.
    @param atm: Definition of atmospheric conditions, pressure and 
                temperature in mb and degrees Celsius respectively
    @param qc:  Input sheath flow in lpm
    @param qm:  Output sheath flow in lpm
    @param DMA: Definition of DMA constants, length (l), inner radius (ri) 
                and outer radius (ro) in meters
    @param V:   Column voltage in Volts
    @return:    zero at correct D for given conditions
    """
    
    #Convert flow rates from lpm to m3/s
    qc = float(qc)/60*0.001
    qm = float(qm)/60*0.001
    
    # Get the instrument constant from the DMA  definition
    gamma = DMA['l']/log((DMA['ro']/DMA['ri']))
    
    # Particle electic mobility
    z = Z(D,atm['T'],atm['P'])
    
    # Central mobility
    zc = (qc+qm)/(4*pi*gamma*V)
    
    return z-zc
    

def ndistr(dp,n=1,T=20):
    '''
    Bipolar charge distribution.
    For particles smaller than a  micron, uses Wiedensohler (1988) 
    but with coefficients corrected for +1 and +2 charges.
    For larger particles, uses Gunn (1956).
    @param dp: diameter of particle in nm
    @param n: number of charges
    @param T: temperature in degree C
    @return: charging efficiency
    @source: Wiedensohler (1988), J. Aerosol Sci., 19, 3.
    @source: Gunn (1956), J. Colloid Sci., 11, 661.
    '''
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
            a0=-0.0003
            a1=-0.1014
            a2=0.3073
            a3=-0.3372
            a4=0.1023
            a5=-0.0105
        elif (n==1):
            a0=-2.3484
            a1=0.6044
            a2=0.4800
            a3=0.0013
            a4=-0.1553
            a5=0.0320
        elif (n==2):
            a0=-44.4756
            a1=79.3772
            a2=-62.8900
            a3=26.4492
            a4=-5.7480
            a5=0.5049
        return 10**(a0+a1*log10(dp)**1+a2*log10(dp)**2/
        +a3*log10(dp)**3+a4*log10(dp)**4+a5*log10(dp)**5)
    else:
        T+=273.15       #convert [°C] to [K]
        dp*=1e-9         #convert [nm] to [m]
        ionconcrat=1     # ratio of positive and negative ion concentrations
        ionmobrat=0.875  # ratio of positive and negative ion mobilities
        f1=e/sqrt(4*pi**2*eps0*dp*kb*T)
        f2=2*pi*eps0*dp*k*T/e**2
        return f1*exp(-1*(n-f2*log(ionconcrat*ionmobrat))**2/(2*f2))
    
