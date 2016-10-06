

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