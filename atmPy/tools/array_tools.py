__author__ = 'htelg'

import numpy as np

def find_closest(array, value):
    """Finds the element of an array which is the closest to a given number and returns its index

    Arguments
    ---------
    array:    array
        The array to search thru.
    value:    float or array-like.
        Number (list of numbers) to search for.

    Return
    ------
    integer or array
        position of closest value(s)"""

    if type(value).__name__ in ('float', 'int', 'float64', 'int64'):
        return np.abs(array - value).argmin()

    elif type(value).__name__ in ('list', 'ndarray'):
        out = np.zeros((len(value)), dtype=int)
        for e, i in enumerate(value):
            out[e] = np.abs(array - i).argmin()
        return out
    else:
        raise ValueError('float,int,array or list are ok types for value. You provided %s' % (type(value).__name__))
