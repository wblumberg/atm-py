import warnings as _warnings
import numpy as _np

moments = {'log normal': ['dNdlogDp', 'dSdlogDp', 'dVdlogDp'],
             'natural': ['dNdDp', 'dSdDp', 'dVdDp'],
             'number': ['dNdlogDp', 'dNdDp'],
             'surface': ['dSdlogDp', 'dSdDp'],
             'volume': ['dVdlogDp', 'dVdDp']}

def convert(dist, to_type, verbose=False):

    dist = dist.copy()
    from_type = dist.distributionType
    if from_type == to_type:
        if verbose:
            _warnings.warn(
                'Distribution type is already %s. Output is an unchanged copy of the distribution' % to_type)
        return dist

    if from_type == 'numberConcentration':
        pass
    elif to_type == 'numberConcentration':
        pass
    elif from_type in moments['log normal']:
        if to_type in moments['log normal']:
            if verbose:
                print('both log normal')
        else:
            dist.data = dist.data / _normal2log(dist)

    elif from_type in moments['natural']:
        if to_type in moments['natural']:
            if verbose:
                print('both natural')
        else:
            dist.data = dist.data * _normal2log(dist)
    else:
        raise ValueError('%s is not an option' % to_type)

    if from_type == 'numberConcentration':
        pass

    elif to_type == 'numberConcentration':
        pass
    elif from_type in moments['number']:
        if to_type in moments['number']:
            if verbose:
                print('both number')
        else:
            if to_type in moments['surface']:
                dist.data *= _2Surface(dist)
            elif to_type in moments['volume']:
                dist.data *= _2Volume(dist)
            else:
                raise ValueError('%s is not an option' % to_type)

    elif from_type in moments['surface']:
        if to_type in moments['surface']:
            if verbose:
                print('both surface')
        else:
            if to_type in moments['number']:
                dist.data /= _2Surface(dist)
            elif to_type in moments['volume']:
                dist.data *= _2Volume(dist) / _2Surface(dist)
            else:
                raise ValueError('%s is not an option' % to_type)

    elif from_type in moments['volume']:
        if to_type in moments['volume']:
            if verbose:
                print('both volume')
        else:
            if to_type in moments['number']:
                dist.data /= _2Volume(dist)
            elif to_type in moments['surface']:
                dist.data *= _2Surface(dist) / _2Volume(dist)
            else:
                raise ValueError('%s is not an option' % to_type)
    else:
        raise ValueError('%s is not an option' % to_type)

    if to_type == 'numberConcentration':
        dist = convert(dist,'dNdDp')
        # dist = dist.convert2dNdDp()
        dist.data *= dist.binwidth

    elif from_type == 'numberConcentration':
        dist.data = dist.data / dist.binwidth
        dist.distributionType = 'dNdDp'
        dist = convert(dist,to_type)
        # dist = dist._convert2otherDistribution(to_type)

    dist.distributionType = to_type
    if verbose:
        print('converted from %s to %s' % (from_type, to_type))
    return dist

def _normal2log(dist):
    trans = (dist.bincenters * _np.log(10.))
    return trans

def _2Surface(dist):
    trans = 4. * _np.pi * (dist.bincenters / 2.) ** 2
    return trans

def _2Volume(dist):
    trans = 4. / 3. * _np.pi * (dist.bincenters / 2.) ** 3
    return trans