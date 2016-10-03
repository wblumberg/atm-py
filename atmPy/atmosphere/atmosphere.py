

def normalize2pressure_and_temperature(data, T_is, T_shall, P_is, P_shall):
    """Normalizes data which is normalized to nomr_is to norm_shall.
    E.g. if you have an as-measured verticle profile of particle concentration,  """
    new_data = data * T_is/T_shall * P_shall/P_is
    return new_data