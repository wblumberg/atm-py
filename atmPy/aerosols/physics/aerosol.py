# -*- coding: utf-8 -*-
from math import pi,exp,log10,sqrt,log

from scipy.optimize import fsolve

from atmPy.general import constants


def z(d, gas, n):
    """
    Calculate electric mobility of particle with diameter D. 
    
    Parameters
    -----------
    d:      float
            diameter in nm
    gas:    object of type gas
            Object that defines the gas calculations
    n:      int
            number of charges
    
    Returns
    ---------   
    Electrical mobility in m2/V*s as defined in Hinds (1999), p. 322, eq. 15.21.
    """
      
    try:
        return n * constants.e * cc(d, gas) / (3 * pi * gas.mu() * d * 1e-9)
    except AttributeError:
        print('Incorrect type selected for attribute "gas".')
        return 0


def z2d(zin, gas, n=1):
    """
    Retrieve particle diameter from the electrical mobility
    
    Call this using a roots or fsolve function.
    
    Parameters
    -----------
    gas:    gas object
            Gas object defining properties of variables related to gases
    zin:    float
            Electrical mobility in m2/Vs
    n:      float, optional, default = 1
            Number of charges
    
    Returns
    -------
    Diameter of particle in nanometers.
    """
    
    # Inline function use with fsolve
    f = lambda d: d*1e-9- n * constants.e * cc(d, gas) / (3 * pi * gas.mu() * zin)
    d0 = 1e-9
    return fsolve(f, d0)[0]


def cc(d, gas):
    """
    Calculate Cunningham correction factor.

    Parameters
    -----------
    d:      float
            Particle diameter in nanometers.
    gas:    Gas object
            gas object from the atmosphere package

    Returns
    --------
    Cunningham correction factor as a function of diameter and mean free path.

    Notes
    -------
    This is from Hinds (1999); p49, eq 3.20
    """
    
    # Convert diameter to microns.
    d = float(d)*1e-3
    # Get the mean free path
    try:

        mfp = gas.l()
        return (1.05*exp(-0.39*d/mfp)+2.34)*mfp/d+1
        
    except AttributeError:
        print('Invalid type entered for "gas".  Should be of type atmosphere.gas".')
        return 0


def kn(dp, gas):
    """
    Calculate the Knudsen number of a particle.

    The Knudsen number determines the appropriateness of the continuum assumption.  If Kn >~1, then the continuum
    assumption is not appropriate for the problem solution.

    Parameters
    ----------
    dp:     float
            particle diameter in nm
    gas:    gas object
            Gas object used to determine the mean free path of the gas.

    Returns
    -------
    float
        Knudsen number

    """
    return 2*gas.l/dp


def ndistr(dp, n=-1, t=20):
    """
    Bipolar charge distribution.

    Parameters
    -----------
    dp: float
        diameter of particle in nm
    n:  int
        number of charges
    t:  float
        temperature in degree C

    Returns
    --------
    Charging efficiency

    Notes
    ------
    * For particles smaller than 1 micron, uses Wiedensohler (1988), J. Aerosol Sci., 19, 3.
    * For particles larger than 1 micron, uses Gunn (1956), J. Colloid Sci., 11, 661.
    """
    
    dp = float(dp)
    a = [0, 0, 0, 0, 0]
    if (abs(n) > 1 and dp < 20) or (dp <= 70 and abs(n) > 2):

        # Particles less than 20 nm can carry at most 1 charge.
        # Particles less than 70 nm can carry at most 2 charges.
        return 0

    # Use Wiedensohler if the particle size is less than a micron and the number of
    # charges is less than or equal to 2.
    elif dp <= 1000 and abs(n) <= 2:
        if n == -2:
            a = [-26.3328, 35.9044, -21.4608, 7.0867, -1.3088, 0.1051]

        elif n == -1:
            a = [-2.3197, 0.6175, 0.6201, -0.1105, -0.1260, 0.0297]
        elif n == 0:
            a = [-0.0003, -0.1014, 0.3073, -0.3372, 0.1023, -0.0105]
        elif n == 1:

            #  a[4] has been modified from the original publication
            a = [-2.3484, 0.6044, 0.4800, 0.0013, -0.1553, 0.0320]
        elif n == 2:

            # a[5] has been modified from original publication
            a = [-44.4756, 79.3772, -62.8900, 26.4492, -5.7480, 0.5049]

        power = 0

        for i, e in enumerate(a):
            power += e*log10(dp)**i

        return 10**power

    # Use Gunn if the particle size is > 1 micron or the number of charges is > 2
    else:

        #  convert [°C] to [K]
        t += 273.15

        # convert [nm] to [m]
        dp *= 1e-9

        # ratio of positive and negative ion concentrations
        ionconcrat = 1

        # ratio of positive and negative ion mobilities
        ionmobrat = 0.875

        f1 = constants.e / sqrt(4 * pi ** 2 * constants.eps0 * dp * constants.k * t)
        f2 = 2*pi * constants.eps0 * dp * constants.k * t / constants.e ** 2
        return f1*exp(-1*(n-f2*log(ionconcrat*ionmobrat))**2/(2*f2))


def d50(n, rhop, q, gas, dj):
    """
    Find the impactor cutpoint.

    Parameters
    ----------
    N:      int
            number of jets
    rhop:   float
            particle density in kg/m^3
    Q:      float
            volumetric flow rate in lpm
    gas:    Gas object
            gas object that utilize a child of the Gas class.
    dj:     int
            jet diameter in meters

    Returns
    -------
    Impactor cutpoint in microns.

    Notes
    ------
    Calculations are those of Hinds (1999)
    
    """

    # Convert the volumetric flow rate into m^3/s
    q = float(q)/60*1000/100**3

    # From Hinds, this is the Stoke's number for the 50% collections
    # efficiency (Table 5.4)
    stk50 = 0.24

    # Equation 5.29 in Hinds (1999)
    d50cc = sqrt(9*gas.mu()*pi*n*dj**3*stk50/(4.0*float(rhop)*q))

    f = lambda x: (d50cc/float(x))**2-cc(float(x*1e-6), gas)
    
    # Find the D50 of the impactor
    return fsolve(f, 0.1)