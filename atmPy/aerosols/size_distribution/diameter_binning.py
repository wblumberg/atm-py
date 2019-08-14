"""
This module contains function that help with dealing with binning of the diameters
in a sizedistribution.
"""

import numpy as np

def bincenters2binsANDnames(bincenters, round=None):
    """This creates bin edges from bincenters

    TODO
    ----
    One might want to distinguisch between logaritmically spaced and linearly spaced
    size distributions.

    Arguments
    ---------
    bincenters: array.
    round: int
        if rounding is desired in the returned column names

    Returns
    -------
    array: binedges
    array: array of strings that can be used as collumn names in sizedistributions
    """
    if type(bincenters) != np.ndarray:
        raise TypeError('Parameter "bincenters" has to be numpy.ndarray. Given object is %s'%type(bincenters))
    noEnds = (bincenters[:-1]+bincenters[1:])/2.
    firstEdge = bincenters[0] - abs(noEnds[0]-bincenters[0])
    lastEdge = bincenters[-1] + abs(noEnds[-1]-bincenters[-1])
    binedges = np.append(firstEdge,noEnds)
    binedges = np.append(binedges,lastEdge)
    minusses = binedges[:-1].copy().astype(str)
    minusses[:] = ' - '
    a = binedges[:-1]
    b = binedges[1:]
    if isinstance(round, int):
        a = a.round(round)
        b = b.round(round)
    a = a.astype(str)
    b = b.astype(str)
    newColnames = np.core.defchararray.add(a,minusses)
    newColnames = np.core.defchararray.add(newColnames,b)
    return binedges,newColnames