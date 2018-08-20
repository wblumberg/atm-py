import os as _os
import pandas as _pd
import numpy as _np
import datetime as _datetime
from atmPy.aerosols.physics import column_optical_properties as _column_optical_properties
from atmPy.general import timeseries as _timeseries

def _path2files(path,
                window,
#                 perform_header_test,
                verbose):
    # folder or single file .... or list
    if _os.path.isdir(path):
        folder = path
        files = _os.listdir(folder)
        files = [f for f in files if 'aer' == f.split('_')[0].split('-')[-1]]
        if verbose:
            print('{} files in folder'.format(len(files)))
    elif _os.path.isfile(path):
        folder, file = _os.path.split(path)
        files = [file]
    else:
        raise ValueError('currently only folder and single files are allowed for the files argument')

    # select sites
#     if site:
#         files = [f for f in files if site in f]
#         if verbose:
#             print('{} files match site specifications.'.format(len(files)))
    # select time window
    if window:
        start, end = window
        files = [f for f in files if (start.replace('-', '') <= f.split('_')[-2] and end.replace('-', '') > f.split('_')[-2])]
        if verbose:
            print('{} of remaining files are in the selected time window.'.format(len(files)))

    # if perform_header_test:
    #     files = [f for f in files if _header_tests(folder, f)]
    #     if verbose:
    #         print('{} of remaining files passed the header test.'.format(len(files)))
    if verbose:
        print('{} number of files will be opened.'.format(len(files)))
    return files, folder

def read_data(folder, fname, header=None):
    """Reads the file takes care of the timestamp and returns a Dataframe
    """
    if not header:
        header = _read_header(folder, fname)

    # dateparse = lambda x: _pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
    df = _pd.read_csv(folder + '/' + fname, skiprows=header['header_size'],
                      sep=',',
#                      delim_whitespace=True,
                     #                      na_values=['N/A'],
                     #                   parse_dates={'times': [0, 1]},
                     #                   date_parser=dateparse
                     )
    df.columns = [c.strip() for c in df.columns]
    df.index = header['start_time'] +  _pd.to_timedelta(df.Stop_UTC, 's')
    df.index.name = 'Time(UTC)'
    df.drop(['Start_UTC', 'Stop_UTC', 'Mid_UTC'], axis=1, inplace=True)
    df.columns = [int(col.split('_')[1]) for col in df.columns]
    df.columns.name = 'AOD@wavelength(nm)'
#     df.rename(columns=_col_label_trans_dict, inplace=True)
    return df

def _read_header(folder, fname):
    """Read the header of file in folder and reterns a dict with relevant data"""
    with open(folder + fname) as rein:
        header_size = int(rein.readline().split(',')[0])
        head = [rein.readline() for e in range(header_size - 2)]
#     return head
    out = {}
    # header size
    out['header_size'] = header_size - 1
    # start time
    out['start_time'] = _pd.to_datetime(_datetime.datetime(*([int(i.strip()) for i in head[5].split(',')][:3])))
    # location
    head[22] = head[22].replace('sea level', 'elevation 0 meters')
    out['location'] = dict([[float(i) if e ==1 else i for e,i in enumerate(li.split()[:2])] for li in head[22].split(':')[1].split(',')])
    # Platform .. equivalent to site
    if head[21].split(':')[0] != 'PLATFORM':
        raise KeyError('We expected to find "PLATFORM" here, found {} instead'.format(head[21].split(':')[0]))
    out['platform'] = head[21].split(':')[1].strip()
    _site_trans = {'Boulder Atmospheric Observatory, Erie, CO': 'BAO',
                  'Smith Point, TX': 'SP'}
    out['abbriviation'] = _site_trans[out['platform']]
    return out
# folder, fname = '/Volumes/HTelg_4TB_Backup/GRAD/mobile/DISCOVER_AQ/SMITH_POINT/','discoveraq-SURFRAD-aer_GROUND-SMITH-POINT_20130829_R0.ict'
# head = _read_header(folder, fname)

# head[22] = head[22].replace('sea level', 'elevation 0 meters')
# head[22]

def _read_files(folder, files, verbose):
    if len(files) == 0:
        raise ValueError('no Files to open')

    if verbose:
        print('Reading files:')
    data_list = []
    header_first = _read_header(folder, files[0])
    for fname in files:
        if verbose:
            print('\t{}'.format(fname), end=' ... ')
        header = _read_header(folder, fname)
        # make sure that all the headers are identical
        if header_first['platform'] != header['platform']:
            raise ValueError('The site name changed from {} to {}!'.format(header_first['platform'], header['platform']))
        data = read_data(folder, fname, header=header)
        data_list.append(data)
        if verbose:
            print('done')

    # concatinate and sort Dataframes and create Timeseries instance
    data = _pd.concat(data_list, sort=True)
    data[data == -999.0] = _np.nan
    data[data == -9.999] = _np.nan
    data = _timeseries.TimeSeries(data, sampling_period=1 * 60)
    data.header = header_first

    if verbose:
        print('done')
    return data

class Mobile_AOD(_column_optical_properties.AOD_AOT):
    pass

def open_path(path = '/Volumes/HTelg_4TB_Backup/GRAD/mobile/DISCOVER_AQ/FRAPPE_BOA/',
              window = ('2014-07-02', '2014-07-04'),
              verbose = True,
              fill_gaps= False,
              keep_original_data = False):


    files, folder = _path2files(path, window, verbose)

    data = _read_files(folder, files, verbose)

    if fill_gaps:
        if verbose:
            print('filling gaps', end=' ... ')
        data.data_structure.fill_gaps_with(what=_np.nan, inplace=True)
        if verbose:
            print('done')

    # get site data
    lon = data.header['location']['longitude']
    lat = data.header['location']['latitude']
    alt = data.header['location']['elevation']
    site_name = data.header['platform']
    abb = data.header['abbriviation']

    # generate Surfrad_aod and add AOD to class
    saod = Mobile_AOD(lat, lon, alt, name=site_name, name_short=abb)

    if keep_original_data:
        saod.original_data = data

    ## add the resulting Timeseries to the class
    data.data.sort_index(axis = 1, inplace=True)
    saod.AOD = data
    return saod