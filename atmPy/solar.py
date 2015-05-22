__author__ = 'mrichardson'

from math import fmod, sin, pi, cos, asin

from atmPy.constants import a2r, r2a

__julian = {"day": 0., "cent": 0.}


class solar(object):

    def __init__(self, ltime):

        julian = solar.juliandates(ltime)
        self.__jday = julian["day"]
        self.__jcent = julian["cent"]

        self.lon = 0
        self.lat = 0

    sinrad = lambda x: sin(a2r(x))
    cosrad = lambda x: cos(a2r(x))

    def juliandates(self, ltime):
        """
        Calculate a Julian date for a given local time

        Parameters
        ----------
        ltime:      float
                    Local time calculated as seconds since Jan 1 1904
        Returns
        --------
        dictionary
            Returns a dictionary of two floats containing julian day and century.
        """
        # Julian day is the continuous count of days since the beginning of the Julian period.
        self.__jday = ltime/(3600*24)+1462+2415018.5
        self.__jcent = (self.__jday-2451545)/36525
        return None

    def __oblelip(self):

        return ((21.448-self.__jcent*(self.__jcent*
                                     (0.00059-(self.__jcent*0.001813))+46.815))/60+26)/60+23

    def __gemeanlon(self):
        return fmod((self.__jcent*0.0003032+36000.76983)*self.__jcent+280.46646, 360)

    def __meananom(self):
        return self.__jcent*(self.__jcent*0.0001537-35999.05029)+357.52911

    def __eartheccen(self):
        return self.__jcent*(self.__jcent*1.267e-7+4.2037e-5)-0.016708634

    def __centsun(self):

        f = lambda x: sin(a2r(x))

        a = f(3)*0.000289
        b = f(2)*(0.019993-self.__jcent*0.000101)
        c = f(1)*(self.__jcent*(self.__jcent**1.45e-5+0.004817)-1.914602)

        return a+b+c

    def __oblcorr(self):
        return self.cosrad(self.__jcent*1934.136-125.04)*0.00256+self.__oblelip()

    def __truelon(self):
        return self.__gemeanlon() + self.__centsun()

    def __app(self):
        a = self.__truelon()-0.00569
        a -= self.sinrad(self.__jcent*1934.136-125.04)*0.00478
        return a

    def __declang(self):
        return r2a(asin(self.sinrad(self.__oblcorr())*self.sinrad(self.__app())))

    def __eq_time(self):
        return None
