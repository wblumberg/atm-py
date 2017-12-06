
class Nephelometer(object):
    def __init__(self):
        #         self._absorbtion_coeff = None
        self._scattering_coeff = None
        self._hemisphericbackscatt_coeff = None
        self._RH_in_instrument = None



    @property
    def scattering_coeff(self):
        return self._scattering_coeff

    @scattering_coeff.setter
    def scattering_coeff(self, value):
        self._scattering_coeff = value

    @property
    def hemisphericbackscatt_coeff(self):
        return self._hemisphericbackscatt_coeff

    @hemisphericbackscatt_coeff.setter
    def hemisphericbackscatt_coeff(self, value):
        self._hemisphericbackscatt_coeff = value

    @property
    def RH_in_instrument(self):
        return self._RH_in_instrument

    @RH_in_instrument.setter
    def RH_in_instrument(self, value):
        self._RH_in_instrument = value

class TandemNephelometer(object):
    def __init__(self):
        self._nephelometer_dry = None
        self._nephelometer_wet = None
