__author__ = 'htelg'


def find_closest(array, value):
    """Finds the element of an array which is the closest to a given number and returns its index

    Arguments
    ---------
    array:    array
        The array to search thru.
    value:    float
        The number to search for.

    Return
    ------
    integer
        position of closest value"""
    return np.abs(array - value).argmin()