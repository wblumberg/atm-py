from atmPy.general import timeseries as _timeseries
from atmPy.general import flightpath as _flightpath

class BalloonSounding(object):
    def __init__(self, data, column_lat='Lat', column_lon='Lon', column_altitude='Altitude'):
        self.timeseries = _timeseries.TimeSeries(data)
        self.vertical_profile = self.timeseries.convert2verticalprofile()
        self.flight_path = _flightpath.FlightPath(self.timeseries, column_lat=column_lat, column_lon=column_lon, column_altitude=column_altitude)