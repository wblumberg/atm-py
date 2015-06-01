import pandas as pd
from atmPy.tools import thermodynamics
from atmPy import timeseries


def read_csv(fname):
    df = pd.read_csv(fname, sep='\t')
    df.index = pd.Series(pd.to_datetime(df.DateTime, format='%Y-%m-%d %H:%M:%S'))
    hk = MantaPayload(df)
    return hk


class MantaPayload(timeseries.TimeSeries):
    def calculate_height(self, p0=1000):
        height = thermodynamics.p2h(self.data.PRESS,
                                    T=273 + self.data.PROBT,
                                    P0=p0
                                    )
        self.data['Height'] = height
        return