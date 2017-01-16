import pandas as _pd
# from atmPy.tools import thermodynamics
from atmPy.general import timeseries as _timeseries
import numpy as _np
from atmPy.aerosols.physics import sampling_efficiency as _sampeff
from atmPy.tools import pandas_tools as _pandas_tools

_date_time_alts = ['uas_datetime']
_pressure_alt = ['StaticP', 'PRESS']
_temp_alt = ['AT_cont', 'AT']
_RH_alt = ['RH_cont', 'RH']
_temp_payload_alt = ['CONDT']
_cn_concentration_alt = ['CONCN']

def read_csv(fname, temperature_limits=(-20, -0.5)):
    """
    Arguments
    ---------
    temerature_limits: tuple.
        The temperature reading has false readings in it which can cause porblems later"""
    df = _pd.read_csv(fname, sep='\t')



    _pandas_tools.ensure_column_exists(df, 'DateTime', _date_time_alts)
    _pandas_tools.ensure_column_exists(df, 'Pressure_Pa', _pressure_alt)
    _pandas_tools.ensure_column_exists(df, 'Temperature', _temp_alt)
    _pandas_tools.ensure_column_exists(df, 'Relative_humidity', _RH_alt)
    _pandas_tools.ensure_column_exists(df, 'Temperature_instrument', _temp_payload_alt, raise_error=False)
    _pandas_tools.ensure_column_exists(df, 'CN_concentration', _cn_concentration_alt)
    try:
        # df.Temperature_payload = df.Temperature_payload.astype(float)
        df.Temperature_instrument = _pd.to_numeric(df.Temperature_instrument, errors='coerce')
        df.CN_concentration = _pd.to_numeric(df.CN_concentration, errors='coerce')

        df.CONCN = _pd.to_numeric(df.CONCN, errors='coerce')
        df.COUNT = _pd.to_numeric(df.COUNT, errors='coerce')
    except AttributeError:
        pass

    # return df
    df.index = _pd.Series(_pd.to_datetime(df.DateTime, format='%Y-%m-%d %H:%M:%S'))


    df = df.drop('DateTime', axis=1)

    df = df.sort_index()

    if temperature_limits:
        df = df[df.Temperature > temperature_limits[0]]
        df = df[temperature_limits[1] > df.Temperature]


    hk = _timeseries.TimeSeries(df)
    hk._data_period = 2
    return hk

# class MantaPayload(timeseries.TimeSeries):

def sample_efficiency(particle_diameters = _np.logspace(_np.log10(0.14), _np.log10(2.5),100),
                            manta_speed = 30, # m/s
                            pressure = 67., #kPa
                            main_inlet_diameter = 4.65 * 1e-3,
                            pick_off_diameter = 2.15 * 1e-3,
                            pick_off_flow_rate = 3,
                            lfe_diameter = 0.7 * 1e-3,
                            verbose = False):

    """Returns the manta sample efficiency for the POPS instruments (up most inlet)

    Parameters
    ----------
    particle_diameters:  float or ndarray
        Particle diameter in um.
    manta_speed: float
        speed of the aircraft in m/s.
    pressure:   flaot
        Barometeric pressure in kPa.
    main_inlet_diameter = 4.65 * 1e-3,
    pick_off_diameter = 2.15 * 1e-3,
    pick_off_flow_rate = 3,
    lfe_diameter = 0.7 * 1e-3,
    verbose = False
    """



    main_inlet_bent = _sampeff.loss_in_a_bent_section_of_circular_tubing(pressure = pressure,  # kPa
                                                  particle_diameter = particle_diameters,  # µm
                                                  tube_air_velocity = manta_speed,  # m/s
                                                  tube_diameter = main_inlet_diameter,  # m
                                                  angle_of_bend = 90,  # degrees
                                                  flow_type = 'auto',
                                                                         verbose = False)

    t_pick_of = _sampeff.loss_in_a_T_junction(particle_diameter=particle_diameters,
                                              particle_velocity=30,
                                              pick_of_tube_diameter=pick_off_diameter,
                                              verbose=False)

    laminar_flow_element = _sampeff.loss_at_an_abrupt_contraction_in_circular_tubing(pressure=pressure,  # kPa
                                                         particle_diameter=particle_diameters,  # µm
                                                         tube_air_velocity=False,  # m/s
                                                         flow_rate_in_inlet=pick_off_flow_rate,  # cc/s
                                                         tube_diameter=pick_off_diameter,  # m
                                                         contraction_diameter=lfe_diameter,  # m
                                                         contraction_angle=90,  # degrees
                                                         verbose=False,
                                                                                     )

    bent_before_pops = _sampeff.loss_in_a_bent_section_of_circular_tubing(
                                                  pressure = pressure,         # kPa
                                                  particle_diameter = particle_diameters,  # µm
                                                  tube_air_velocity = False, # m/s
                                                  tube_air_flow_rate = pick_off_flow_rate,
                                                  tube_diameter = pick_off_diameter,   # m
                                                  angle_of_bend = 90,       # degrees
                                                  flow_type = 'auto',
                                                  verbose = False)

    gravitational_loss = _sampeff.gravitational_loss_in_circular_tube(pressure=101.3,  # kPa
                                            particle_diameter=particle_diameters,  # µm
                                            tube_diameter=pick_off_diameter,  # m
                                            tube_length=0.25,  # m
                                            incline_angle=0,  # degrees from horizontal (0-90)
                                            flow_rate=3,  # cc/s
                                            mean_flow_velocity=False,  # 0.1061    # m/s)
                                            flow_type='auto',
                                                                      verbose=False)


    loss_list = [main_inlet_bent, t_pick_of, laminar_flow_element, bent_before_pops, gravitational_loss]
    names  = ['all_losses', 'main_inlet_bent', 't_pick_of', 'laminar_flow_element', 'bent_before_pops', 'gravitational_loss']

    all_losses = 1
    for l in loss_list:
        all_losses *= l

    loss_list.insert(0,all_losses)



    df = _pd.DataFrame(_np.array(loss_list).transpose(), columns = names, index =particle_diameters * 1e3)
    df.index.name = 'diameters_nm'
    return df