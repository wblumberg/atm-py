# -*- coding: utf-8 -*-
"""
This module contains atmospheric constands and standards.

@author: Hagen
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def standard_atmosphere(value, quantity='altitude', standard='international', return_standard=False):
    """Returns pressure, temperature, and/or altitude as a function of pressure, or altitude for the standard international atmosphere

    Arguments
    ---------
    value: float or ndarray.
        Depending on the keyword "quantity" this is:
        - altitude in meters.
        - pressure in mbar.

    quantity: 'altitude' or 'pressure'.
        quantaty of the argument value.

    standard: 'US' or 'international'.
        defines which standard is used.

    return_standard: bool, optional.
        if True argument "value" and "quantity" are ignored and a pandas dataTable with the standard is returned.

    Returns
    -------
    tuple of two floats or two ndarrays depending on type of h:
        First quantaty the tuple is pressure in mbar or altitude in meter, second is temperatur in Kelvin.



    """
    if standard == 'international':
        alt = np.array([-610, 11000, 20000, 32000, 47000, 51000, 71000, 84852]).astype(float)
        pressure = np.array([108900, 22632, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734]) / 100.
        tmp = np.array([19, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.28]) + 273.15

    elif standard == 'US':
        alt = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000]).astype(float)
        pressure = np.array([101325, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642]) / 100.
        tmp = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])

    else:
        raise TypeError('No standard with the name "%s" is defined' % standard)

    if return_standard:
        return pd.DataFrame(np.array([alt, pressure, tmp]).transpose(),
                            columns=['Altitude_meter', 'Pressure_mbar', 'Temperature_K'])

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
