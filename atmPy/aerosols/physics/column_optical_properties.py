from atmPy.general import measurement_site as _measurement_site
import pandas as _pd
import numpy as _np
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

        self.site = _measurement_site.Station(lat, lon, elevation, name=name, abbreviation=name_short, info = site_info)


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

    @property
    def ang_exp(self):
        return self._ang_exp

    @ang_exp.setter
    def ang_exp(self, value):
        self._ang_exp = value

    def aod2angstrom_exponent(self, column_1=500, column_2=870, wavelength_1=None, wavelength_2=None):
        if wavelength_1 == None:
            wavelength_1 = column_1
        if wavelength_2 == None:
            wavelength_2 = column_2
        c1 = column_1
        c2 = column_2
        c1ex = wavelength_1
        c2ex = wavelength_2
        out = - _np.log10(self.AOD.data.loc[:, c1] / self.AOD.data.loc[:, c2]) / _np.log10(c1ex / c2ex)
        out = _timeseries.TimeSeries(_pd.DataFrame(out))
        return out

