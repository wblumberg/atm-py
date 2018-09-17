import numpy as _np
import pandas as _pd
import os as _os
from atmPy.general import timeseries as _timeseries
from atmPy.aerosols.physics import column_optical_properties as _column_optical_properties
# from atmPy.general import measurement_site as _measurement_site
# from atmPy.radiation import solar as _solar

locations = [{'name': 'Bondville',
              'state' :'IL',
              'abbriviations': ['BND', 'bon'],
              'lon': -88.37309,
              'lat': 40.05192,
              'alt' :230,
              'timezone': -6},
             {'name': 'Sioux Falls',
              'state': 'SD',
              'abbriviations': ['SXF', 'sxf'],
              'lon': -96.62328,
              'lat': 43.73403,
              'alt': 473,
              'timezone': -6},
             {'name': 'Table Mountain',
              'state': 'CO',
              'abbriviations': ['TBL', 'tbl'],
              'lon': -105.23680,
              'lat': 40.12498,
              'alt': 1689,
              'timezone': -7}
             ]

_col_label_trans_dict = {'OD413': 415,
                         'OD414': 415,
                         'OD415': 415,
                         'OD416': 415,
                         'OD495': 500,
                         'OD496': 500,
                         'OD497': 500,
                         'OD499': 500,
                         'OD500': 500,
                         'OD501': 500,
                         'OD502': 500,
                         'OD503': 500,
                         'OD504': 500,
                         'OD505': 500,
                         'OD609': 614,
                         'OD612': 614,
                         'OD614': 614,
                         'OD615': 614,
                         'OD664': 673,
                         'OD670': 673,
                         'OD671': 673,
                         'OD672': 673,
                         'OD673': 673,
                         'OD674': 673,
                         'OD676': 673,
                         'OD861': 870,
                         'OD868': 870,
                         'OD869': 870,
                         'OD870': 870,
                         'OD1623': 1625,
                         'OD1624': 1625,
                         ### sometime AOD was used instead of OD =>
                         'AOD414': 415,
                         'AOD415': 415,
                         'AOD497': 500,
                         'AOD501': 500,
                         'AOD609': 614,
                         'AOD615': 614,
                         'AOD664': 673,
                         'AOD673': 673,
                         'AOD861': 870,
                         'AOD870': 870,
                         }


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
            timezone = [l for l in locations if header['site'] in l['name']][0]['timezone']
            df.index += _pd.to_timedelta(-1 * timezone, 'h')
            df.index.name = 'Time (UTC)'
        else:
            df.index.name = 'Time (local)'
        df.rename(columns=_col_label_trans_dict, inplace=True)
        return df

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
        if header_first['site'] != header['site']:
            raise ValueError('The site name changed from {} to {}!'.format(header_first['site'], header['site']))
        data = read_data(folder, fname, UTC = UTC, header=header)
        data_list.append(data)
        if verbose:
            print('done')

    # concatinate and sort Dataframes and create Timeseries instance
    data = _pd.concat(data_list, sort=True)
    data[data == -999.0] = _np.nan
    data[data == -9.999] = _np.nan
    data = _timeseries.TimeSeries(data, sampling_period=1 * 60)
    data.header = header_first

    if cloud_sceened:
        data.data[data.data['0=good'] == 1] = _np.nan
    if verbose:
        print('done')
    return data

class Surfrad_AOD(_column_optical_properties.AOD_AOT):
    pass
#     def __init__(self, lat, lon, elevation = 0, name = None, name_short = None, timezone = 0):
#         self._aot = None
#         self._aot = None
#         self._sunposition = None
#         self._timezone = timezone
#
#         self.site = _measurement_site.Site(lat, lon, elevation, name=name, abbriviation=name_short)
#
#
#     @property
#     def sun_position(self):
#         if not self._sunposition:
#             if self._timezone != 0:
#                 date = self.AOD.data.index +  _pd.to_timedelta(-1 * self._timezone, 'h')
#             else:
#                 date = self.AOD.data.index
#             self._sunposition = _solar.get_sun_position(self.site.lat, self.site.lon, date)
#             self._sunposition.index = self.AOD.data.index
#             self._sunposition = _timeseries.TimeSeries(self._sunposition)
#         return self._sunposition
#
#     @property
#     def AOT(self):
#         if not self._aot:
#             if not self._aod:
#                 raise AttributeError('Make sure either AOD or AOT is set.')
#             aot = self.AOD.data.mul(self.sun_position.data.airmass, axis='rows')
#             aot.columns.name = 'AOT@wavelength(nm)'
#             aot = _timeseries.TimeSeries(aot)
#             self._aot = aot
#         return self._aot
#
#     @ AOT.setter
#     def AOT(self,value):
#         self._aot = value
#
#     @property
#     def AOD(self):
#         if not self._aod:
#             if not self._aot:
#                 raise AttributeError('Make sure either AOD or AOT is set.')
#             aod = self.AOT.data.dif(self.sun_position.data.airmass, axis='rows')
#             aod.columns.name = 'AOD@wavelength(nm)'
#             aod = _timeseries.TimeSeries(aod)
#             self._aod = aod
#         return self._aod
#
#     @ AOD.setter
#     def AOD(self,value):
#         self._aod = value



def open_path(path = '/Volumes/HTelg_4TB_Backup/SURFRAD/aftp/aod/bon',
              site = 'bon',
              window = ('2017-01-01', '2017-01-02'),
              cloud_sceened = True,
              local2UTC = False,
              perform_header_test = False,
              verbose = False,
              fill_gaps= False,
              keep_original_data = False):

    if site:
        if len([loc for loc in locations if site in loc['abbriviations']]) == 0:
            raise ValueError('The site {} has not been set up yet. Add relevant data to the location dictionary'.format(site))

    files, folder = _path2files(path, site, window, perform_header_test, verbose)
    data = _read_files(folder, files, verbose, UTC=local2UTC, cloud_sceened=cloud_sceened)

    if fill_gaps:
        if verbose:
            print('filling gaps', end=' ... ')
        data.data_structure.fill_gaps_with(what=_np.nan, inplace=True)
        if verbose:
            print('done')

    # add Site class to surfrad_aod
    try:
        site = [l for l in locations if data.header['site'] in l['name']][0]
    except IndexError:
        raise ValueError('Looks like the site you trying to open is not set up correctly yet in "location"')

    lon = site['lon']
    lat = site['lat']
    alt = site['alt']
    timezone = site['timezone']
    site_name = site['name']
    abb = site['abbriviations'][0]
    # saod.site = _measurement_site.Site(lat, lon, alt, name=site_name, abbriviation=abb)

    # generate Surfrad_aod and add AOD to class
    saod = Surfrad_AOD(lat, lon, alt, name=site_name, name_short=abb, timezone = timezone, site_info = site)
    if keep_original_data:
        saod.original_data = data

    if local2UTC:
        saod._timezone = 0
    else:
        saod._timezone = timezone
    ## select columns that show AOD
    data_aod = data._del_all_columns_but(_np.unique(_np.array(list(_col_label_trans_dict.values()))))

    ## rename columns
    data_aod.data.columns.name = 'AOD@wavelength(nm)'
    data_aod.data.sort_index(axis = 1, inplace=True)

    ## add the resulting Timeseries to the class
    saod.AOD = data_aod

    saod.ang_exp = data._del_all_columns_but('Ang_exp')
    return saod