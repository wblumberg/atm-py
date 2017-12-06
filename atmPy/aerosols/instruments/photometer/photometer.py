class Photometer(object):
    def __init__(self):
        self._absorbtion_coeff = None

    @property
    def absorbtion_coeff(self):
        return self._absorbtion_coeff

    @absorbtion_coeff.setter
    def absorbtion_coeff(self, value):
        self._absorbtion_coeff = value