from netCDF4 import Dataset as _Dataset
import os as _os
from atmPy.data_archives.arm import _tdmasize,_tdmaapssize,_tdmahyg,_aosacsm, _noaaaos
import pandas as _pd

# arm_products = {'tdmasize':   {'read': _tdmasize._parse_netCDF,    'concat': _tdmasize._concat_rules},
#                 'tdmaapssize':{'read': _tdmaapssize._parse_netCDF, 'concat': _tdmaapssize._concat_rules},
#                 'tdmahyg':    {'read': _tdmahyg._parse_netCDF,     'concat': _tdmahyg._concat_rules}
#               }

arm_products = {'tdmasize':   {'module': _tdmasize},
                'tdmaapssize':{'module': _tdmaapssize},
                'tdmahyg':    {'module': _tdmahyg},
                'aosacsm':    {'module': _aosacsm},
                'noaaaos':    {'module': _noaaaos}
                }



def read_cdf(fname,
             data_product = None,
             time_window = None,
             concat = True,
             ignore_unknown = False,
             verbose = True,
             read_only = None):
    """
    Reads ARM NetCDF file(s) and returns a containers with the results.

    Parameters
    ----------
    fname: str or list of str.
        Either a file, directory, or list of files. If directory name is given
        all files in the directory will be considered.
    data_product: str.
        To see a list of allowed products look at the variable arm_products.
    time_window: tuple of str.
        e.g. ('2016-01-25 15:22:40','2016-01-29 15:00:00').
        Currently the entire day is considered, no need to use exact times.
    concat
    ignore_unknown
    verbose
    read_only

    Returns
    -------

    """

    # list or single file
    if type(fname) == str:
        fname = [fname]

    if type(data_product) == str:
        data_product = [data_product]
    products = {}

    #loop throuh files
    for f in fname:
        if verbose:
            print('\n', f)

        # error handling: test for netCDF file format
        if _os.path.splitext(f)[-1] != '.cdf':
            txt = '\t %s is not a netCDF file ... skipping'%f
            if verbose:
                print(txt)
            continue

        fnt = _os.path.split(f)[-1].split('.')

        # check if in time_window
        if time_window:
            ts = fnt[-3]
            file_start_data = _pd.to_datetime(ts)
            start_time = _pd.to_datetime(time_window[0])
            end_time = _pd.to_datetime(time_window[1])
            dt_start = file_start_data - start_time
            dt_end = file_start_data - end_time

            if dt_start.total_seconds() < -86399:
                if verbose:
                    print('outside (before) the time window ... skip')
                continue
            elif dt_end.total_seconds() > 86399:
                if verbose:
                    print('outside (after) the time window ... skip')
                continue

        foundit = False
        for prod in arm_products.keys():
            if prod in fnt[0]:
                product_id = prod
                foundit = True
                break

        # test if unwanted product
        if data_product:
            if product_id not in data_product:
                if verbose:
                    print('Not the desired data product ... skip')
                continue

        if not foundit:
            txt = '\t has no ncattr named platform_id. Guess from file name failed ... skip'
            if verbose:
                print(txt)
            continue

        elif read_only:
            if product_id not in read_only:
                if verbose:
                    print('not in read_only')
                continue




        # Error handling: if product_id not in products
        if product_id not in arm_products.keys():
            txt = 'Platform id %s is unknown.'%product_id
            if ignore_unknown:
                if verbose:
                    print(txt + '... skipping')
                continue
            else:
                raise KeyError(txt)

        if product_id not in products.keys():
            products[product_id] = []


        file_obj = _Dataset(f)
        out = arm_products[product_id]['module']._parse_netCDF(file_obj)
        file_obj.close()
        products[product_id].append(out)



    if len(fname) == 1:
        return out

    else:
        if concat:
            for pf in products.keys():
                products[pf] = arm_products[pf]['module']._concat_rules(products[pf])
        return products




################