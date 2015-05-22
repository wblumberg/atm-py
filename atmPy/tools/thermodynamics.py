import numpy as np
import scipy.constants as const
from hagpack import unit_conversion


def h2p(h, T=293., P0=1000., m=28.966, unit_p='mbar'):
    """
    parameters:
        h:      altitude from reference higth in meter (if reference hight is not sealevel, you probably want to adjust P0)
    optional parameters:
        T:     Temperature in K
        P0:    Pressure at reference altitude in hPa (default = 1000.)
        m:     average mass of gas molecules in u (default = 28.966)
        unit_p: either 'mbar' or 'torr'
    source:
        http://en.wikipedia.org/wiki/Barometric_formula
    """
    k = const.physical_constants['Boltzmann constant'][0]
    g = const.physical_constants['standard acceleration of gravity'][0]
    # m *= 1.66053892e-27
    m *= 1 / const.physical_constants['Avogadro constant'][0] / 1000.
    Ph = P0 * np.exp((-1 * m * g * h) / (k * T))
    return Ph


def p2h(p, T=293., P0=1000., m=28.966, unit_p='mbar'):
    """
    parameters:
        h:      altitude from reference higth in meter (if reference hight is not sealevel, you probably want to adjust P0)
    optional parameters:
        T:     Temperature in K
        P0:    Pressure at reference altitude in hPa (default = 1000.)
        m:     average mass of gas molecules in u (default = 28.966)
        unit_p: either 'mbar' or 'torr'
    source:
        http://en.wikipedia.org/wiki/Barometric_formula
    """
    if unit_p == 'torr':
        p = unit_conversion.torr2mbar(p)
    k = const.physical_constants['Boltzmann constant'][0]
    g = const.physical_constants['standard acceleration of gravity'][0]
    m *= 1 / const.physical_constants['Avogadro constant'][0] / 1000.
    h = (np.log(P0) - np.log(p)) * ((k * T) / (m * g))
    return h