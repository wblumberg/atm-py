from atmPy.general import measurement_site as _measurement_site
from atmPy.general import timeseries as _timeseries
import numpy as _np
import os as _os
import pandas as _pd
from atmPy.aerosols.physics import column_optical_properties as _column_optical_properties


def _read_header(folder, fname):
    """Read the header of file in folder and reterns a dict with relevant data"""
    with open(folder + '/' + fname) as myfile:
        head = [next(myfile) for x in range(6)]

    out = {}
    # version
    out['version'] = head[0].replace(';', '').split()[-1]
    if out['version'] == '3':
        out['header_size'] = 6
    else:
        raise ValueError('Version {} unknown. Programming required')

    # level
    out['level'] = head[2].split(':')[1].split()[-1]
    # product
    out['product'] = head[2].split(':')[1].split()[0]
    # site
    out['site'] = head[1].strip()

    return out


def _header_tests(folder, fname, version='3', level='2.0', data_product='AOD', site='BONDVILLE', raise_error=True):
    """This tests if the file is actually holding what it promisses"""
    with open(folder + '/' + fname) as myfile:
        head = [next(myfile) for x in range(6)]

    # version correct?
    isversion = head[0].replace(';', '').split()[-1] == version
    # level correct ?
    islevel = head[2].split(':')[1].split()[-1] == level
    # product correct?
    isproduct = head[2].split(':')[1].split()[0] == data_product
    # site correct?
    site_from_fname = head[1].strip()
    issite = site_from_fname == site

    if raise_error:
        assert (isversion)
        assert (islevel)
        assert (isproduct)
        if not issite:
            raise KeyError('{} does not macht the site requirement of {}'.format(site_from_fname, site))

    return _np.all(_np.array([isversion, islevel, isproduct, issite]))


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
        files = [f for f in files if
                 (start.replace('-', '') < f.split('_')[1] and end.replace('-', '') > f.split('_')[0])]
        if verbose:
            print('{} of remaining files are in the selected time window.'.format(len(files)))

    if perform_header_test:
        files = [f for f in files if _header_tests(folder, f, site = site)]
        if verbose:
            print('{} of remaining files passed the header test.'.format(len(files)))
    return files, folder


def _read_files(folder, files, verbose):
    def read_data(folder, filename, header=None):
        """Reads the file takes care of the timestamp and returns a Dataframe
        """
        if not header:
            header = _read_header(folder, filename)
        dateparse = lambda x: _pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
        df = _pd.read_csv(folder + '/' + filename, skiprows=header['header_size'],
                          #                      na_values=['N/A'],
                          parse_dates={'times': [0, 1]},
                          date_parser=dateparse)

        df = df.set_index('times')

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
        data = read_data(folder, fname, header=header)
        data_list.append(data)
        if verbose:
            print('done')

    # concatinate and sort Dataframes and create Timeseries instance
    data = _pd.concat(data_list)
    data[data == -999.0] = _np.nan
    data = _timeseries.TimeSeries(data, sampling_period=15 * 60)
    if verbose:
        print('done')
    return data


class Aeronet_AOD(_column_optical_properties.AOD_AOT):
    pass


def open_path(path='/Volumes/HTelg_4TB_Backup/AERONET/',
              site='BONDVILLE',
              window=('2017-01-01', '2017-02-01'),
              perform_header_test=True,
              fill_gaps=False,
              verbose=False,):

    files, folder = _path2files(path, site, window, perform_header_test, verbose)
    #     files = clean_up_files(files, None, None, None, verbose)
    data = _read_files(folder, files, verbose)

    if fill_gaps:
        if verbose:
            print('filling gaps', end=' ... ')
        data.data_structure.fill_gaps_with(what=_np.nan, inplace=True)
        if verbose:
            print('done')

    if window:
        data = data.zoom_time(window[0], window[1])
    if data.data.shape[0] == 0:
        raise ValueError('There is no data in the selected time window.')
    # add Site class to Aeronet_AOD class
    lon = data.data['Site_Longitude(Degrees)'].dropna().iloc[0]
    lat = data.data['Site_Latitude(Degrees)'].dropna().iloc[0]
    alt = data.data['Site_Elevation(m)'].dropna().iloc[0]
    site_name = data.data['AERONET_Site_Name'].dropna().iloc[0]

    # aaod.site = _measurement_site.Site(lat, lon, alt, name=site_name)
    # generate Aeronet_AOD and add AOD to class
    aaot = Aeronet_AOD(lat, lon, elevation = alt, name = site_name, name_short = None, timezone = 0)

    ## select columns that show AOD
    aodcols = [col for col in data.data.columns if (col[:3] == 'AOD' and 'Empty' not in col)]
    data_aot = data._del_all_columns_but(aodcols)
    newcol = _np.array([c.replace('AOD_', '').replace('nm', '') for c in aodcols]).astype(int)

    ## rename columns
    data_aot.data.rename(columns=dict(zip(aodcols, newcol)), inplace=True)
    data_aot.data.columns.name = 'AOD@wavelength(nm)'
    data_aot.data.dropna(axis=1, how='all', inplace=True)

    ## add the resulting Timeseries to the class
    aaot.AOD = data_aot

    return aaot