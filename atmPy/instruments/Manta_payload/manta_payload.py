import pandas as pd
from atmPy.tools import thermodynamics
from atmPy import timeseries


def read_csv(fname, temperature_limits=(-20, -0.5)):
    """
    Arguments
    ---------
    temerature_limits: tuple.
        The temperature reading has false readings in it which can cause porblems later"""
    df = pd.read_csv(fname, sep='\t')
    df.index = pd.Series(pd.to_datetime(df.DateTime, format='%Y-%m-%d %H:%M:%S'))
    df['Pressure_Pa'] = df.PRESS
    df['Temperature'] = df.AT
    df['Relative_humidity'] = df.RH
    df = df.drop('PRESS', axis=1)
    df = df.drop('AT', axis=1)
    df = df.drop('RH', axis=1)

    df = df.sort_index()

    if temperature_limits:
        df = df[df.Temperature > temperature_limits[0]]
        df = df[temperature_limits[1] > df.Temperature]

    hk = timeseries.TimeSeries(df)
    return hk

# class MantaPayload(timeseries.TimeSeries):
