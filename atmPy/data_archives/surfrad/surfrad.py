import numpy as _np
import pandas as _pd
import os as _os
from atmPy.general import timeseries as _timeseries
from atmPy.general import measurement_site as _measurement_site

locations = [{'name': 'Bondville',
              'state' :'IL',
              'abbriviations': ['BND', 'bon'],
              'lon': -88.37309,
              'lat': 40.05192,
              'alt' :230,
              'timezone': 6}]

def _path2files(path, site, window, perform_header_test, verbose):
    # folder or single file .... or list
    if _os.path.isdir(path):
        folder = path
        files = _os.listdir(folder)
        if verbose:
            print('{} files in folder'.format(len(files)))
    elif _os.path.isfile(path):
        folder, file = _os.path.split(path)
        files = [file]
    else:
        raise ValueError('currently only folder and single files are allowed for the files argument')

    # select sites
    if site:
        files = [f for f in files if site in f]
        if verbose:
            print('{} files match site specifications.'.format(len(files)))
    # select time window
    if window:
        start, end = window
        files = [f for f in files if (start.replace('-', '') <= f.split('_')[1].split('.')[0] and end.replace('-', '') > f.split('_')[1].split('.')[0])]
        if verbose:
            print('{} of remaining files are in the selected time window.'.format(len(files)))

    # if perform_header_test:
    #     files = [f for f in files if _header_tests(folder, f)]
    #     if verbose:
    #         print('{} of remaining files passed the header test.'.format(len(files)))
    return files, folder

def _read_header(folder, fname):
    """Read the header of file in folder and reterns a dict with relevant data"""
    header_size = 5
    with open(folder + '/' + fname) as myfile:
        head = [next(myfile) for x in range(header_size)]

    out = {}
    # header size
    out['header_size'] = header_size
    # site
    out['site'] = head[0].split()[0]
    # channels
    channels = head[2].split()
    out['channels'] = channels[:channels.index('channel')]

    # date
    out['date'] = _pd.to_datetime(head[1].split()[0])
    #     return head
    return out

def _read_files(folder, files, verbose, UTC = False, cloud_sceened = True):
    def read_data(folder, fname, UTC = False, header=None):
        """Reads the file takes care of the timestamp and returns a Dataframe
        """
        if not header:
            header = _read_header(folder, fname)

        # dateparse = lambda x: _pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
        df = _pd.read_csv(folder + '/' + fname, skiprows=header['header_size'],
                         delim_whitespace=True,
                         #                      na_values=['N/A'],
                         #                   parse_dates={'times': [0, 1]},
                         #                   date_parser=dateparse
                         )

        datetimestr = '{0:0>4}{1:0>2}{2:0>2}'.format(header['date'].year, header['date'].month, header['date'].day)+ df.ltime.apply \
            (lambda x: '{0:0>4}'.format(x)) + 'UTC'  # '+0000'
        df.index = _pd.to_datetime(datetimestr, format="%Y%m%d%H%M%Z")
        if UTC:
            timezone = [l for l in locations if header['site'] in l['abbriviations']][0]['timezone']
            df.index += _pd.to_timedelta(timezone, 'h')
            df.index.name = 'Time (UTC)'
        else:
            df.index.name = 'Time (local)'
        return df

    if verbose:
        print('Reading files:')
    data_list = []
    header_first = _read_header(folder, files[0])
    for fname in files:
        if verbose:
            print('\t{}'.format(fname), end=' ... ')
        header = _read_header(folder, fname)
        # make sure that all the headers are identical
        assert (header_first == header)
        data = read_data(folder, fname, UTC = UTC, header=header)
        data_list.append(data)
        if verbose:
            print('done')

    # concatinate and sort Dataframes and create Timeseries instance
    data = _pd.concat(data_list)
    data[data == -999.0] = _np.nan
    data = _timeseries.TimeSeries(data, sampling_period=1 * 60)
    data.header = header_first

    if cloud_sceened:
        data.data[data.data['0=good'] == 1] = _np.nan
    if verbose:
        print('done')
    return data

class Surfrad_AOD(object):
    pass

def open_path(path = '/Volumes/HTelg_4TB_Backup/SURFRAD/aftp/aod/bon/2017',
              site = 'bon',
              window = ('2017-01-01', '2017-01-02'),
              cloud_sceened = True,
              local2UTC = False,
              perform_header_test = False,
              verbose = False,
              fill_gaps= False):


    files, folder = _path2files(path, site, window, perform_header_test, verbose)

    data = _read_files(folder, files, verbose, UTC=local2UTC, cloud_sceened=cloud_sceened)

    if fill_gaps:
        if verbose:
            print('filling gaps', end=' ... ')
        data.data_structure.fill_gaps_with(what=_np.nan, inplace=True)
        if verbose:
            print('done')

    # generate Surfrad_aod and add AOD to class
    saod = Surfrad_AOD()

    ## select columns that show AOD
    aodcols = [col for col in data.data.columns if 'OD' in col]
    data_aod = data._del_all_columns_but(aodcols)
    aodcols.sort(key = lambda x: int(x.replace('OD' ,'')))

    newcol = _np.array(data.header['channels']).astype(float)
    newcol.sort()

    # test if something will go  wrong with the renaming
    aodcolstest = _np.array([int(c.replace('OD' ,'')) for c in aodcols])

    if _np.any((aodcolstest - newcol) > 1):
        raise ValueError('Something went wrong with the renaming of the labels ... programming required')

    ## rename columns
    data_aod.data.rename(columns=dict(zip(aodcols, newcol)), inplace=True)
    data_aod.data.columns.name = 'AOD@wavelength(nm)'
    data_aod.data.sort_index(axis = 1, inplace=True)
    # data_aod.data.dropna(axis=1, how='all', inplace=True)

    ## add the resulting Timeseries to the class

    saod.AOD = data_aod

    # add Site class to surfrad_aod
    site = [l for l in locations if data.header['site'] in l['abbriviations']][0]
    lon = site['lon']
    lat = site['lat']
    alt = site['alt']
    site_name = site['name']
    abb = site['abbriviations'][0]
    saod.site = _measurement_site.Site(lat, lon, alt, name=site_name, abbriviation=abb)

    return saod