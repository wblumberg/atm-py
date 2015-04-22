import pandas as pd
from atmPy.tools import time_tools
from atmPy import housekeeping


def _drop_some_columns(data):
    data.drop('Clock', axis=1, inplace=True)
    data.drop('Year', axis=1, inplace=True)
    data.drop('Month', axis=1, inplace=True)
    data.drop('Day', axis=1, inplace=True)
    data.drop('Hours', axis=1, inplace=True)
    data.drop('Minutes', axis=1, inplace=True)
    data.drop('Seconds', axis=1, inplace=True)


def read_file(fname):
    picof = open(fname, 'r')
    header = picof.readline()
    picof.close()

    header = header.split(' ')
    header_cleaned = []

    for head in header:
        bla = head.replace('<', '').replace('>', '')
        where = bla.find('[')
        if where != -1:
            bla = bla[:where]
        header_cleaned.append(bla)

    data = pd.read_csv(fname,
                       names=header_cleaned,
                       sep=' ',
                       skiprows=1,
                       header=0)

    data.drop(range(20), inplace=True)  # dropping the first x lines, since the time is often dwrong

    timeseries = data.Year.astype(str) + '-' + data.Month.astype(str) + '-' + data.Day.astype(
        str) + ' ' + data.Hours.apply(lambda x: '%02i' % x) + ':' + data.Minutes.apply(
        lambda x: '%02i' % x) + ':' + data.Seconds.apply(lambda x: '%05.2f' % x)
    data.index = pd.Series(pd.to_datetime(timeseries, format=time_tools.get_time_formate()))

    _drop_some_columns(data)

    # convert from rad to deg
    data.Lat.values[:] = np.rad2deg(data.Lat.values)
    data.Lon.values[:] = np.rad2deg(data.Lon.values)

    return housekeeping.HouseKeeping(data, {'original header': header})

#
# class AutoPilot(object):
# def __init__(self, data, info):
# self.data = data
# self.info = info
