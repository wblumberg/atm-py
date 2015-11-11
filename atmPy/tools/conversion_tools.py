import numpy as np
import scipy.constants as const
from hagpack import unit_conversion


def h2p(h, T=293., P0=1000., m=28.966, unit_p='mbar'):
    """
    Parameters
    ----------
    h: {flaot, array}
        altitude from reference higth in meter (if reference hight is not sealevel, you probably want to adjust P0)
    T: float, optional
        Temperature in K
    P0: float, optional
        Pressure at reference altitude in hPa (default = 1000.)
    m: float, optional
        average mass of gas molecules in u (default = 28.966)
    unit_p: {[mbar], torr}, optional

    Source
    ------
    http://en.wikipedia.org/wiki/Barometric_formula
    """

    k = const.physical_constants['Boltzmann constant'][0]
    g = const.physical_constants['standard acceleration of gravity'][0]
    m *= 1 / const.physical_constants['Avogadro constant'][0] / 1000.
    p = P0 * np.exp((-1 * m * g * h) / (k * T))

    if unit_p == 'torr':
        p = unit_conversion.mbar2torr(p)
    return p


def p2h(p, T=293., P0=1000., m=28.966, unit_p='mbar'):
    """ Returns an elevation from barometric pressure

    Parameters
    ----------
    p: {float, array}
        barometric pressure in mbar or torr specified with unit_p
    T: float, optional
        Temperature in K
    P0: float, optional
        Pressure at reference altitude in hPa (default = 1000.)
    m: float, optional
        average mass of gas molecules in u (default = 28.966)
    unit_p: {[mbar], torr}, optional

    Source
    ------
    http://en.wikipedia.org/wiki/Barometric_formula
    """

    if unit_p == 'torr':
        p = unit_conversion.torr2mbar(p)
    k = const.physical_constants['Boltzmann constant'][0]
    g = const.physical_constants['standard acceleration of gravity'][0]
    m *= 1 / const.physical_constants['Avogadro constant'][0] / 1000.
    h = (np.log(P0) - np.log(p)) * ((k * T) / (m * g))
    return h