# -*- coding: utf-8 -*-
"""
This module contains atmospheric constands and standards.

@author: Hagen
"""

import numpy as np
from scipy.interpolate import interp1d


def standard_atmosphere_international(value, quantity='altitude'):
    """Returns pressure, temperature, and/or altitude as a function of pressure, or altitude for the standard international atmosphere

    Arguments
    ---------
    value: float or ndarray.
        Depending on the keyword "quantity" this is:
        - altitude in meters.
        - pressure in mbar.

    quantity: 'altitude' or 'pressure'.
        quantaty of the argument value.

    Returns
    -------
    tuple of two floats or two ndarrays depending on type of h:
        First quantaty in the first tuple is pressure in mbar or altitude in meter, second is temperatur in Kelvin.
    """

    alt = np.array([-610, 11000, 20000, 32000, 47000, 51000, 71000, 84852]).astype(float)
    pressure = np.array([108900, 22632, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734]) / 100.
    tmp = np.array([19, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.28]) + 273.15

    if quantity == 'altitude':
        pressure_int = interp1d(alt, np.log(pressure), kind='cubic')
        press_n = np.exp(pressure_int(value))
        out = press_n

    elif quantity == 'pressure':
        alt_int = interp1d(np.log(pressure), alt, kind='cubic')
        alt_n = alt_int(np.log(value))
        out = alt_n
        value = alt_n
    else:
        raise TypeError('Quantity "$s$" is not an option' % quantity)

    tmp_int = interp1d(alt, tmp, kind='linear')
    tmp_n = tmp_int(value)
    return out, tmp_n
