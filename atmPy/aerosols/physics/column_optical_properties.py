from atmPy.general import measurement_site as _measurement_site
import pandas as _pd
# import numpy as _np
from atmPy.radiation import solar as _solar
from atmPy.general import timeseries as _timeseries


class AOD_AOT(object):
    def __init__(self, lat, lon, elevation = 0, name = None, name_short = None, timezone = 0, site_info = None):
        """This class is for column AOD or AOT meausrements at a fixed site. This class is most usfull for aerosol
        optical properties from a CIMEL (AERONET) or a MFRSR (SURFRAD)
        Parameters
        ----------
        lat, lon: location of site
            lat: deg north, lon: deg east
            e.g. lat = 40.05192, lon = -88.37309 for Bondville, IL, USA
        elevation: float
            elevation of site in meter (not very importent parameter, keeping it at zero is usually fine ...
            even vor Boulder
        name, name_short: str
            name and abbriviation of name for site
        timezone: int
            Timezon in houres, e.g. Bondville has the timezone -6.
            Note, this can lead to some confusion if the data is in UTC not in local time.... this needs to be improved
            ad some point"""

        self._aot = None
        self._aod = None
        self._sunposition = None
        self._timezone = timezone

        self.site = _measurement_site.Site(lat, lon, elevation, name=name, abbriviation=name_short, info = site_info)


    @property
    def sun_position(self):
        if not self._sunposition:
            if self._timezone != 0:
                date = self._timestamp_index +  _pd.to_timedelta(-1 * self._timezone, 'h')
            else:
                date = self._timestamp_index
            self._sunposition = _solar.get_sun_position(self.site.lat, self.site.lon, date)
            self._sunposition.index = self._timestamp_index
            self._sunposition = _timeseries.TimeSeries(self._sunposition)
        return self._sunposition

    @property
    def AOT(self):
        if not self._aot:
            if not self._aod:
                raise AttributeError('Make sure either AOD or AOT is set.')
            aot = self.AOD.data.mul(self.sun_position.data.airmass, axis='rows')
            aot.columns.name = 'AOT@wavelength(nm)'
            aot = _timeseries.TimeSeries(aot)
            self._aot = aot
        return self._aot

    @ AOT.setter
    def AOT(self,value):
        self._aot = value
        self._aot.data.columns.name = 'AOT@wavelength(nm)'
        self._timestamp_index = self._aot.data.index

    @property
    def AOD(self):
        if not self._aod:
            if not self._aot:
                raise AttributeError('Make sure either AOD or AOT is set.')
            aod = self.AOT.data.div(self.sun_position.data.airmass, axis='rows')
            aod.columns.name = 'AOD@wavelength(nm)'
            aod = _timeseries.TimeSeries(aod)
            self._aod = aod
        return self._aod

    @ AOD.setter
    def AOD(self,value):
        self._aod = value
        self._aod.data.columns.name = 'AOD@wavelength(nm)'
        self._timestamp_index = self._aod.data.index