import scipy as _sp
import numpy as _np

def normalize2pressure_and_temperature(data, P_is, P_shall, T_is, T_shall):
    """Normalizes data which is normalized to nomr_is to norm_shall.
    E.g. if you have an as-measured verticle profile of particle concentration

    Parameters
    ----------
    data: int, float, ndarray, pandas.DataFrame ....
        the data
    T_is: int, float, ndarray, pandas.DataFrame ...
        Temp which it is currently normalized to, e.g. instrument temperature.
    T_shall: int, float, ndarray, pandas.DataFrame ...
        Temp to normalize to, e.g. standard temperature.
    P_is: int, float, ndarray, pandas.DataFrame ...
        Pressure which it is currently normalized to, e.g. instrument Pressure.
    P_shall: int, float, ndarray, pandas.DataFrame ...
        Pressure to normalize to, e.g. standard Pressure."""

    new_data = data * T_is/T_shall * P_shall/P_is
    return new_data

def normalize2standard_pressure_and_temperature(data, P_is, T_is):
    out = normalize2pressure_and_temperature(data, P_is, 1000 , T_is, 273.15)
    return out


class Barometric_Formula(object):
    def __init__(self,
                 pressure=None,
                 pressure_ref=None,
                 temp=None,
                 alt=None,
                 alt_ref=None,
                 laps_rate = 0.0065):
        self._pressure = pressure
        self._pressure_ref = pressure_ref
        self._temp = temp
        self._alt = alt
        self._alt_ref = alt_ref
        self._R = _sp.constants.gas_constant
        self._g = _sp.constants.g
        self._M = 0.0289644  # kg/mol
        self._L = laps_rate # K/m

    #         self._dict = {'pressure': self._pressure,
    #                       'pressure_ref': self._pressure_ref,
    #                       'temperature': self._temp,
    #                       'altitude': self._alt,
    #                       'altitude_ref': self._alt_ref}

    @property
    def altitude(self):
        if not self._alt:
            self._alt = self._get_alt()
        return self._alt

    @altitude.setter
    def altitude(self, value):
        self._alt = value

    @property
    def altitude_ref(self):
        if not self._alt_ref:
            pass
        return self._alt_ref

    @altitude_ref.setter
    def altitude_ref(self, value):
        self._alt_ref = value

    @property
    def pressure(self):
        if not self._pressure:
            pass
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        self._pressure = value

    @property
    def pressure_ref(self):
        if not self._pressure_ref:
            pass
        return self._pressure_ref

    @pressure_ref.setter
    def pressure_ref(self, value):
        self._pressure_ref = value

    @property
    def temperature(self):
        if not self._temp:
            pass
        return self._temp

    @temperature.setter
    def temperature(self, value):
        self._temp = value


    def _get_alt(self):
        self._check_var('altitude')
        alt = (self._temp + 273.15) / self._L * ((self._pressure_ref / self._pressure)**((self._R * self._L) / (self._g * self._M)) - 1)
        return alt

    def _check_var(self, attr, all_but=True):
        trans_dict = {'pressure': self._pressure,
                      'pressure_ref': self._pressure_ref,
                      'temperature': self._temp,
                      'altitude': self._alt,
                      'altitude_ref': self._alt_ref}
        if all_but:
            req = list(trans_dict.keys())
            req.pop(req.index(attr))
        else:
            req = attr
        missing = []
        for var in req:
            if type(trans_dict[var]) == type(None):
                missing.append(var)
        if len(missing) == 0:
            return
        elif len(missing) == 1:
            txt = 'Make sure you assign {}.'.format(*missing)
        else:
            txt = 'The following attributes need to be set before this can work:\n\t'
            print()
            txt += '\n\t'.join(missing)
        self.missing = missing
        raise AttributeError(txt)