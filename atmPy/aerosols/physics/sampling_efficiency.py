# all functions are based on http://aerosols.wustl.edu/AAARworkshop08/software/AEROCALC-11-3-03.xls

import warnings

import numpy as np
import pandas as pd
from atmPy.aerosols.physics import _tools_sampling_efficiency


def loss_in_a_T_junction(temperature=293.15,
                         pressure=65.3,
                         particle_diameter=2.5,
                         particle_velocity=30,
                         particle_density=1000,
                         pick_of_tube_diameter=2.15 * 1e-3,
                         verbose=False):
    """Returns the fraction of particles which make from a main tubing into a T-like pick-of based on the stopping distancde

    Arguments
    ---------
    temperature: float.
        Temperature in Kelvin.
    pressure: float.
        pressure in kPa.
    particle_diameter: float.
        in meter
    particle_velocity: float.
        in meter/second.
    particle_density: float.
        kg/m^3
    verbose: bool.
        """
    diag_os = np.sqrt(2) * pick_of_tube_diameter
    diag_is = pick_of_tube_diameter
    mid_dig = (diag_os + diag_is) / 2
    effective_tube_diameter = mid_dig / np.sqrt(2)

    if not hasattr(particle_diameter, '__iter__'):
        particle_diameter = [particle_diameter]

    df = pd.DataFrame()
    for d in particle_diameter:

        pl = _tools_sampling_efficiency.stopping_distance(temperature=temperature,
                                                          pressure=pressure,
                                                          particle_diameter=d,
                                                          particle_velocity=particle_velocity,
                                                          particle_density=particle_density,
                                                          verbose=verbose)


        out = 1. - pl / effective_tube_diameter

        if not hasattr(out, '__iter__'):
            out = [out]
        df[d] = out

    df[df < 0] = 0

    if verbose:
        print('loss_in_a_T_junction: %s' % df)
    return df


def loss_at_an_abrupt_contraction_in_circular_tubing(temperature=293.15,  # Kelvin
                                                     pressure=101.3,  # kPa
                                                     particle_diameter=1,  # µm
                                                     particle_density=1000,  # kg/m^3
                                                     tube_air_velocity=False,  # m/s
                                                     flow_rate_in_inlet=3,  # cc/s
                                                     tube_diameter=0.0025,  # m
                                                     contraction_diameter=0.00125,  # m
                                                     contraction_angle=90,  # degrees
                                                     verbose=False,
                                                     ):
    """
    (B&W 8-69 to 8-71; W&B 6-54, 17-25)
    Temperature	293.15	 Kelvin
    Pressure	101.3	 kPa
    Particle diameter	1	 µm
    Particle density	1000	 kg/m^3
    Tube air velocity	10	 m/s
    Tube diameter	0.0025	 m
    Contraction diameter	0.00125	 m
    Contraction angle	90	 degrees
    """

    if not tube_air_velocity:
        tube_air_velocity = _tools_sampling_efficiency.flow_rate2flow_velocity(flow_rate_in_inlet, tube_diameter, verbose=verbose)

    st_num = _tools_sampling_efficiency.stokes_number(particle_density, particle_diameter, pressure, temperature, tube_air_velocity, 1,
                                                      contraction_diameter, verbose=verbose)


    # st_num = (particle_density * particle_diameter * particle_diameter * 0.000000000001 * slip_correction_factor * B906 / (18*B912*B908))


    frac = 1 - (1 / (1 + ((2 * st_num * (1 - (contraction_diameter / tube_diameter) ** 2)) / (
    3.14 * np.exp(-0.0185 * contraction_angle))) ** -1.24))
    return frac


# def aspiration_efficiency_all_forward_angles(temperature=293.15,  # Kelvin
#                                              pressure=101.3,  # kPa
#                                              particle_diameter=10,  # µm
#                                              particle_density=1000,  # kg/m^3
#                                              inlet_diameter=0.025,  # m
#                                              sampling_angle=46,  # degrees	between 0 to 90°
#                                              flow_rate_in_inlet=3,  # cc/s
#                                              air_velocity_in_inlet=False,  # m/s
#                                              velocity_ratio=5,  # R is 1 for isokinetic, > 1 for subisokinetic
#                                              force=False,
#                                              verbose=False):
#     """
#     (B&W 8-20, 8-21, 8-22; W&B 6-20, 6-21, 6-22)
#     Hangal and Willeke Eviron. Sci. Tech. 24:688-691 (1990)
#     Temperature	293.15	 Kelvin
#     Pressure	101.3	 kPa
#     Particle diameter	10	 µm
#     Particle density	1000	 kg/m^3
#     Inlet diameter	0.025	 m
#     Sampling angle	46	 degrees	between 0 to 90°
#     Air velocity in inlet (Vi)	0.34	 m/s
#     Velocity ratio (Vw/Vi)	5		R is 1 for isokinetic, > 1 for subisokinetic
#     """
#     if not air_velocity_in_inlet:
#         air_velocity_in_inlet = _tools_sampling_efficiency.flow_rate2flow_velocity(flow_rate_in_inlet, inlet_diameter, verbose=verbose)
#
#     st_num = _tools_sampling_efficiency.stokes_number(particle_density, particle_diameter, pressure, temperature, air_velocity_in_inlet,
#                                                       velocity_ratio, inlet_diameter, verbose=verbose)
#     # rey_num = tools.flow_reynolds_number(inlet_diameter, air_velocity_in_inlet, temperature, pressure, verbose=verbose)
#     if np.any(45 < sampling_angle) and np.any(sampling_angle <= 90) \
#             and np.any(1.25 < velocity_ratio) and np.any(velocity_ratio < 6.25) \
#             and np.any(0.003 < st_num) and np.any(st_num < 0.2):
#         pass
#     else:
#         txt = """sampling angle, velocity ratio, or stokes number is not in valid regime!
# Sampling angle: %s (45 < angle < 90)
# Velocity ratio: %s (1.25 < ratio < 6.25)
# Stokes number: %s (0.003 < st_num  < 0.2)""" % (sampling_angle, velocity_ratio, st_num)
#         if force:
#             warnings.warn(txt)
#         else:
#             raise ValueError(txt)
#
#     if sampling_angle == 0:
#         inert_param = 1 - 1 / (1 + (2 + 0.617 / velocity_ratio) * st_num)
#     elif sampling_angle > 45:
#         inert_param = 3 * st_num ** (velocity_ratio ** -0.5)
#     else:
#         f619 = st_num * np.exp(0.022 * sampling_angle)
#         inert_param = (1 - 1 / (1 + (2 + 0.617 / velocity_ratio) * f619)) * (
#         1 - 1 / (1 + 0.55 * f619 * np.exp(0.25 * f619))) / (1 - 1 / (1 + 2.617 * f619))
#     # =IF(B609=0,1-1/(1+(2+0.617/B611)*B618),IF(B609>45,3*B618^(B611^-0.5),(1-1/(1+(2+0.617/B611)*B619))*(1-1/(1+0.55*B619*EXP(0.25*B619)))/(1-1/(1+2.617*B619))))
#
#
#     asp_eff = 1 + (velocity_ratio * np.cos(sampling_angle * np.pi / 180) - 1) * inert_param
#     return asp_eff




def loss_in_a_bent_section_of_circular_tubing(temperature=293.15,  # Kelvin
                                              pressure=101.3,  # kPa
                                              particle_diameter=3.5,  # µm
                                              particle_density=1000,  # kg/m^3
                                              tube_air_flow_rate=3,  # cc/s
                                              tube_air_velocity=False,  # m/s
                                              tube_diameter=0.0025,  # m
                                              angle_of_bend=90,  # degrees
                                              flow_type='auto',
                                              verbose=False
                                              ):
    """ (B&W 8-66 to 8-68; W&B 6-52, 6-53)
        Temperature	293.15	 Kelvin
        Pressure	101.3	 kPa
        Particle Diameter	3.5	 µm
        Particle Density	1000	 kg/m^3
        Tube air velocity	6.25	 m/s
        Tube diameter	0.0025	 m
        Angle of bend	90	 degrees"""

    if not tube_air_velocity:
        tube_air_velocity = _tools_sampling_efficiency.flow_rate2flow_velocity(tube_air_flow_rate, tube_diameter, verbose=verbose)

    if flow_type == 'auto':
        flow_type = _tools_sampling_efficiency.test_flow_type_in_tube(tube_diameter, tube_air_velocity, temperature, pressure, verbose=verbose)

    velocity_ratio = 1
    stnum = _tools_sampling_efficiency.stokes_number(particle_density, particle_diameter, pressure, temperature, tube_air_velocity,
                                                     velocity_ratio, tube_diameter, verbose=verbose)

    if flow_type == 'laminar':

        fract = 1 - stnum * angle_of_bend * np.pi / 180.

    elif flow_type == 'turbulent':
        fract = np.exp(-2.823 * stnum * angle_of_bend * np.pi / 180)

    else:
        raise ValueError('Unknown flow type: %s' % flow_type)

    return fract


def gravitational_loss_in_circular_tube(temperature=293.15,  # Kelvin
                                        pressure=101.3,  # kPa
                                        particle_diameter=10,  # µm
                                        particle_density=1000,  # kg/m^3
                                        tube_diameter=0.01,  # m
                                        tube_length=0.1,  # m
                                        incline_angle=60,  # degrees from horizontal (0-90)
                                        flow_rate=3,  # cc/s
                                        mean_flow_velocity=False,  # 0.1061    # m/s)
                                        flow_type='auto',
                                        verbose=False):
    """
    Arguments
    ---------
    temperature         = 293.15,   # Kelvin
    pressure            = 101.3,    # kPa
    particle_diameter   = 10,       # µm
    particle_density    = 1000,     # kg/m^3
    tube_diameter       = 0.01,     # m
    tube_length         = 0.1,      # m
    incline_angle       = 60,       # degrees from horizontal (0-90)
    flow_rate           = 3,        # cc/s
    mean_flow_velocity  = False     #0.1061    # m/s)"""

    if not mean_flow_velocity:
        mean_flow_velocity = _tools_sampling_efficiency.flow_rate2flow_velocity(flow_rate, tube_diameter, verbose=verbose)

    if flow_type == 'auto':
        flow_type = _tools_sampling_efficiency.test_flow_type_in_tube(tube_diameter, mean_flow_velocity, temperature, pressure, verbose=verbose)

    if flow_type == 'laminar':
        sv = _tools_sampling_efficiency.settling_velocity(temperature, particle_density, particle_diameter, pressure, verbose=verbose)
        k770 = np.cos(np.pi * incline_angle / 180) * 3 * sv * tube_length / (4 * tube_diameter * mean_flow_velocity)
        if np.any((k770 ** (2. / 3)) > 1):
            fract = 0
        else:

            if np.any(k770 ** (2 / 3.) > 1):
                k771 = 0
            else:
                k771 = np.arcsin(k770 ** (1 / 3.))  # done

            fract = (1 - (2 / np.pi) * (
            2 * k770 * np.sqrt(1 - k770 ** (2 / 3)) + k771 - (k770 ** (1 / 3) * np.sqrt(1 - k770 ** (2 / 3)))))  # done
            if np.any(fract < 0):
                fract = 0
    elif flow_type == 'turbulent':
        raise ValueError('Sorry this loss mechanism has not been implemented for turbulent flow')
    else:
        raise ValueError('Unknown flow type: %s' % flow_type)
    if verbose:
        print('k770: %s' % k770)
        print('k771: %s' % k771)
        print('fraction penetrating: %s' % fract)
    return fract


# def gravitational_loss_in_an_inlet(temperature=293.15,  # K
#                                    pressure=101.3,  # hPa ... kPa?!?
#                                    particle_diameter=15,  # um
#                                    particle_density=1000,  # kg/m^3
#                                    inlet_diameter=0.0127,  # m
#                                    inlet_length=1.,  # m
#                                    sampling_angle=45,  # deg; keep between 0 to 90°
#                                    air_velocity_in_inlet=30,  # m/s;
#                                    velocity_ratio=1.5,
#                                    # R is 1 for isokinetic, > 1 for subisokinetic, < 1 for superisokinetic
#                                    verbose=False):
#     """Gravitational losses in an inlet (B&W 8-23, 8-24; W&B 6-23, 6-24)
#     Not to be mixed up with gravitational loss in a circular tube
#
#     Arguments
#     ---------
#     temperature:    float.
#             Temperature in K.
#     pressure:       float.
#             Pressure in kPa.
#     particle_diameter:  float.
#             Aerosol particle diameter in micro meter.
#     particle_density:   float.
#             Density of the particle material in kg/m^3.
#     inlet_diameter:     float.
#             Inlet diameter in m.
#     inlet_length:       float.
#             Inlent length in m.
#     sampling_angle:     float.
#             Angle of the inlet in deg. 0 is horizontal; keep between 0 to 90°.
#     air_velocity_in_inlet:  float.
#             Velocity of the air in inlet in m/s.
#     velocity_ratio:         float.
#             Ratio between velocity outside and inside the inlet. R is 1 for isokinetic, > 1 for subisokinetic, < 1 for superisokinetic
#     verbose: bool.
#         if results are printed.
#     """
#
#     out = np.exp(-4.7 * _tools_sampling_efficiency.K(sampling_angle, inlet_length, air_velocity_in_inlet, inlet_diameter, particle_density,
#                                                      particle_diameter, pressure, temperature, velocity_ratio, verbose=verbose) ** 0.75)
#     if verbose:
#         print('Fraction lost due to gravitation: %s' % out)
#     return out


def inlet_efficiency_isoaxial_horizontal_sharp_edged(temperature = 293.15,      # Kelvin
                                                     pressure = 101.3,          # kPa
                                                     particle_diameter = 15,    # µm
                                                     particle_density = 1000,   # kg/m^3
                                                     inlet_diameter = 0.0127,   # m
                                                     inlet_length = 0.1,        # m
                                                     sampling_flow_rate = 3,    # cc/s
                                                     air_velocity_inlet = False,# m/s
                                                     ambient_air_speed =  25.5, # m/s
                                                     velocity_ratio = False,       # R is 1 for isokinetic, > 1 for subisokinetic
                                                     verbose = False
                                                     ):
    """
    Inlet efficiency for an isoaxial horizontal sharp-edged inlet: aspiration + transmission (B&W 8-8, 8-14, 8-16, 8-18; W&B 6-8, 6-14, 6-16, 6-18, Hinds 10-7)
    Temperature	293.15	 Kelvin
    Pressure	101.3	 kPa
    Particle diameter	15	 µm
    Particle density	1000	 kg/m^3
    Inlet diameter	0.0127	 m
    Inlet length	0.1	 m
    Air velocity in inlet (Vi)	0.05	 m/s
    Velocity ratio (Vw/Vi)	15

    Parameters
    ----------
    temperature
    pressure
    particle_diameter
    particle_density
    inlet_diameter
    inlet_length
    air_velocity_inlet
    velocity_ratio
    verbose

    Returns
    -------

    """
    if not air_velocity_inlet:
        air_velocity_inlet = _tools_sampling_efficiency.flow_rate2flow_velocity(sampling_flow_rate, inlet_diameter, verbose=verbose)

    if not velocity_ratio:
        velocity_ratio = ambient_air_speed/air_velocity_inlet

    st_no = _tools_sampling_efficiency.stokes_number(particle_density, particle_diameter, pressure, temperature, air_velocity_inlet, velocity_ratio, inlet_diameter, verbose=verbose)
    asp_eff = 1 + (velocity_ratio - 1) * (1 - (1 / (1 + (2 + (0.617 / velocity_ratio)) * st_no)))

    set_vel = _tools_sampling_efficiency.settling_velocity(temperature, particle_density, particle_diameter, pressure, verbose=verbose)

    grav_param =(inlet_length / air_velocity_inlet) / (inlet_diameter / set_vel)
    flow_reyno = _tools_sampling_efficiency.flow_reynolds_number(inlet_diameter, air_velocity_inlet, temperature, pressure, verbose=verbose)
    grav_eff = np.exp(-4.7 * (np.sqrt(grav_param * st_no / np.sqrt(flow_reyno)))**0.75)

    if velocity_ratio > 1:
        init_trans_eff = (1 + (velocity_ratio - 1) / (1 + 2.66 / st_no**(2/3))) / (1 + (velocity_ratio - 1) / (1 + 0.418 / st_no))
    else:
        init_trans_eff = 1

    def vena_contracta_efficiency():
        if velocity_ratio < 1:
            eff = np.exp(-75 * (0.09 * (st_no * (air_velocity_inlet - air_velocity_inlet * velocity_ratio) / (air_velocity_inlet * velocity_ratio))**0.3)**2)
        else:
            eff = 1
        return eff

    vena_cont_eff =  vena_contracta_efficiency()
    efficiency = asp_eff * grav_eff * init_trans_eff * vena_cont_eff
    return efficiency

def test_inlet_efficiency_isoaxial_horizontal_sharp_edged():
    calc = inlet_efficiency_isoaxial_horizontal_sharp_edged(temperature=293.15,
                                                    pressure=101.3,
                                                    particle_diameter=15,
                                                    particle_density=1000,
                                                    inlet_diameter=0.0127,
                                                    inlet_length=0.1,
                                                    air_velocity_inlet=0.05,
                                                    velocity_ratio=15,
                                                    verbose=False)
    if calc != 0.71686004968714068:
        txt = "Test failed for inlet_efficiency_isoaxial_horizontal_sharp_edged"
        raise ValueError(txt)

    else:
        return True
