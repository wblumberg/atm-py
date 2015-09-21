# all functions are based on http://aerosols.wustl.edu/AAARworkshop08/software/AEROCALC-11-3-03.xls

import numpy as np
import warnings

def air_density(temperature, pressure, verbose=False):
    out = 1.293 * (273.15 / temperature) * (pressure / 101.3)
    if verbose:
        print('air density: %s' % out)
    return out


def particle_reynolds_number(temperature,  # Kelvin
                             pressure,  # kPa
                             particle_diameter,  # µm
                             particle_velocity,  # m/s		 (B&W 4-1; W&B 3-1; Hinds 2-41)
                             verbose=False
                             ):
    """
    Temperature	293.15	 Kelvin
    Pressure	101.3	 kPa
    Particle diameter	5	 µm
    Particle velocity	0.01	 m/s

    """
    ad = air_density(temperature, pressure, verbose=verbose)
    av = air_viscosity(temperature, verbose=verbose)
    out = 0.000001 * ad * particle_diameter * particle_velocity / av
    if verbose:
        print('Particle reynolds number: %s' % out)
    return out

def flow_reynolds_number(inlet_diameter, air_velocity_in_inlet, temperature, pressure, verbose=False):
    """definition!"""
    out = air_density(temperature, pressure, verbose=verbose) * inlet_diameter * air_velocity_in_inlet / air_viscosity(
        temperature, verbose=verbose)
    if verbose:
        print('flow reynolds number: %s' % out)
    return out


def gravitational_dep_parameter(inlet_length, temperature, particle_density, particle_diameter, air_velocity_in_inlet,
                                inlet_diameter, verbose=False):
    """what is that???"""
    out = inlet_length * settling_velocity(temperature, particle_density, particle_diameter, verbose=verbose) / (
    air_velocity_in_inlet * inlet_diameter)
    if verbose:
        print('flow reynolds number: %s' % out)
    return out


def stokes_number(particle_density, particle_diameter, pressure, temperature, air_velocity_in_inlet, velocity_ratio,
                  inlet_diameter, verbose=False):
    """what is that"""
    scf = slip_correction_factor(pressure, particle_diameter, verbose=verbose)
    av = air_viscosity(temperature, verbose=verbose)
    out = (particle_density * particle_diameter ** 2 * 0.000000000001 * scf / (
    18 * av)) * air_velocity_in_inlet * velocity_ratio / inlet_diameter
    if verbose:
        print('stokes number: %s' % out)
    return out


def slip_correction_factor(pressure, particle_diameter, verbose=False):
    """define"""
    out = 1 + (2 / (pressure * particle_diameter * 0.752)) * (
    6.32 + 2.01 * np.exp(-0.1095 * pressure * 0.752 * particle_diameter))
    if verbose:
        print('slip_correction_factor: %s' % out)
    return out


def air_viscosity(temperature, verbose=False):
    """define"""
    out = 0.00001708 * ((temperature / 273.15) ** 1.5) * ((393.396) / (temperature + 120.246))
    if verbose:
        print('air_viscosity: %s' % out)
    return out


def settling_velocity(temperature, particle_density, particle_diameter, pressure, verbose=False):
    """define!"""
    out = particle_density * particle_diameter ** 2 * 0.000000000001 * 9.81 * slip_correction_factor(pressure,
                                                                                                     particle_diameter,
                                                                                                     verbose=verbose) / (
          18 * air_viscosity(temperature, verbose=verbose))
    if verbose:
        print('settling_velocity: %s' % out)
    return out


def gravitational_dep_parameter(inlet_length, air_velocity_in_inlet, inlet_diameter, temperature, particle_density,
                                particle_diameter, pressure, verbose=False):
    """what is that???"""
    out = inlet_length * settling_velocity(temperature, particle_density, particle_diameter, pressure,
                                           verbose=verbose) / (air_velocity_in_inlet * inlet_diameter)
    if verbose:
        print('gravitational_dep_parameter: %s' % out)
    return out


def K(sampling_angle, inlet_length, air_velocity_in_inlet, inlet_diameter, particle_density, particle_diameter,
      pressure, temperature, velocity_ratio, verbose=False):
    """what is that?"""
    gdp = gravitational_dep_parameter(inlet_length, air_velocity_in_inlet, inlet_diameter, temperature,
                                      particle_density, particle_diameter, pressure, verbose=verbose)
    sn = stokes_number(particle_density, particle_diameter, pressure, temperature, air_velocity_in_inlet,
                       velocity_ratio, inlet_diameter, verbose=verbose)
    frn = flow_reynolds_number(inlet_diameter, air_velocity_in_inlet, temperature, pressure, verbose=verbose)
    out = (np.sqrt(gdp * sn) * frn ** -0.25) * np.sqrt(np.cos(np.deg2rad(sampling_angle)))
    if verbose:
        print('K: %s' % out)
    return out


def stopping_distance(temperature=293.15,  # Kelvin
                      pressure=65.3,  # kPa
                      particle_diameter=2.5,  # µm
                      particle_velocity=30,  # m/s
                      particle_density=1000,  # kg/m^3
                      verbose=False, ):
    """
        (B&W 4-34, 36; W&B 3-34, 3-36; Hinds 5-19, 20, 21)
        Temperature	293.15	 Kelvin
        Pressure	65.3	 kPa
        Particle diameter	2.5	 µm
        Particle velocity	30	 m/s
        Particle density	1000	 kg/m^3"""

    rey_num = particle_reynolds_number(temperature, pressure, particle_diameter, particle_velocity, verbose=False)
    scf = slip_correction_factor(pressure, particle_diameter, verbose=verbose)
    av = air_viscosity(temperature, verbose=verbose)
    ad = air_density(temperature, pressure, verbose=verbose)
    if np.any(rey_num < 1):
        out = particle_density * particle_velocity * particle_diameter ** 2 * 0.000000000001 * scf / (av * 18)
    else:
        out = particle_density * particle_diameter * 0.000001 * (
        (rey_num) ** (1 / 3) - np.arctan(((rey_num) ** (1 / 3)) / np.sqrt(6)) * np.sqrt(6)) / ad
    # =IF(B261<1,B257*B256*B255*B255*0.000000000001*B262/(B260*18),B257*B255*0.000001*((B261)^(1/3)-ATAN(((B261)^(1/3))/SQRT(6))*SQRT(6))/(B259
    if verbose:
        print('stopping distance: %s m' % out)
    return out


##########################
def flow_rate2flow_velocity(flow_rate, diameter_tubing, verbose=False):
    """
    Parameters
    ----------
    flow_rate: float.
        in cc/s
    diameter_tubing: float.
        in m

    Returns
    -------
    flow velocity in m/s"""
    flow_rate = flow_rate * 1e-6
    vel = 4 * flow_rate / (np.pi * diameter_tubing ** 2)
    if verbose:
        print('mean flow velocity: %s' % vel)
    return vel

###########################
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

    pl = stopping_distance(temperature=temperature,
                           pressure=pressure,
                           particle_diameter=particle_diameter,
                           particle_velocity=particle_velocity,
                           particle_density=particle_density,
                           verbose=verbose)
    out = 1. - pl / pick_of_tube_diameter
    if verbose:
        print('loss_in_a_T_junction: %s' % out)
    return out


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
        tube_air_velocity = flow_rate2flow_velocity(flow_rate_in_inlet, tube_diameter, verbose=verbose)

    st_num = stokes_number(particle_density, particle_diameter, pressure, temperature, tube_air_velocity, 1,
                           contraction_diameter, verbose=verbose)


    # st_num = (particle_density * particle_diameter * particle_diameter * 0.000000000001 * slip_correction_factor * B906 / (18*B912*B908))


    frac = 1 - (1 / (1 + ((2 * st_num * (1 - (contraction_diameter / tube_diameter) ** 2)) / (
    3.14 * np.exp(-0.0185 * contraction_angle))) ** -1.24))
    return frac


def aspiration_efficiency_all_forward_angles(temperature=293.15,  # Kelvin
                                             pressure=101.3,  # kPa
                                             particle_diameter=10,  # µm
                                             particle_density=1000,  # kg/m^3
                                             inlet_diameter=0.025,  # m
                                             sampling_angle=46,  # degrees	between 0 to 90°
                                             flow_rate_in_inlet=3,  # cc/s
                                             air_velocity_in_inlet=False,  # m/s
                                             velocity_ratio=5,  # R is 1 for isokinetic, > 1 for subisokinetic
                                             force=False,
                                             verbose=False):
    """
    (B&W 8-20, 8-21, 8-22; W&B 6-20, 6-21, 6-22)
    Hangal and Willeke Eviron. Sci. Tech. 24:688-691 (1990)
    Temperature	293.15	 Kelvin
    Pressure	101.3	 kPa
    Particle diameter	10	 µm
    Particle density	1000	 kg/m^3
    Inlet diameter	0.025	 m
    Sampling angle	46	 degrees	between 0 to 90°
    Air velocity in inlet (Vi)	0.34	 m/s
    Velocity ratio (Vw/Vi)	5		R is 1 for isokinetic, > 1 for subisokinetic
    """
    if not air_velocity_in_inlet:
        air_velocity_in_inlet = flow_rate2flow_velocity(flow_rate_in_inlet, inlet_diameter, verbose=verbose)

    st_num = stokes_number(particle_density, particle_diameter, pressure, temperature, air_velocity_in_inlet,
                           velocity_ratio, inlet_diameter, verbose=verbose)
    rey_num = flow_reynolds_number(inlet_diameter, air_velocity_in_inlet, temperature, pressure, verbose=verbose)
    if (45 < sampling_angle <= 90) and (1.25 < velocity_ratio < 6.25) and (0.003 < st_num < 0.2):
        pass
    else:
        txt = """sampling angle, velocity ratio, or stokes number is not in valid regime!
Sampling angle: %s (45 < angle < 90)
Velocity ratio: %s (1.25 < ratio < 6.25)
Stokes number: %s (0.003 < st_num  < 0.2)""" % (sampling_angle, velocity_ratio, st_num)
        if force:
            warnings.warn(txt)
        else:
            raise ValueError(txt)

    if sampling_angle == 0:
        inert_param = 1 - 1 / (1 + (2 + 0.617 / velocity_ratio) * st_num)
    elif sampling_angle > 45:
        inert_param = 3 * st_num ** (velocity_ratio ** -0.5)
    else:
        f619 = st_num * np.exp(0.022 * sampling_angle)
        inert_param = (1 - 1 / (1 + (2 + 0.617 / velocity_ratio) * f619)) * (
        1 - 1 / (1 + 0.55 * f619 * np.exp(0.25 * f619))) / (1 - 1 / (1 + 2.617 * f619))
    # =IF(B609=0,1-1/(1+(2+0.617/B611)*B618),IF(B609>45,3*B618^(B611^-0.5),(1-1/(1+(2+0.617/B611)*B619))*(1-1/(1+0.55*B619*EXP(0.25*B619)))/(1-1/(1+2.617*B619))))


    asp_eff = 1 + (velocity_ratio * np.cos(sampling_angle * np.pi / 180) - 1) * inert_param
    return asp_eff


def test_flow_type(tube_diameter, tube_air_velocity, temperature, pressure, verbose=False):
    rey_num = flow_reynolds_number(tube_diameter, tube_air_velocity, temperature, pressure, verbose=verbose)


def loss_in_a_bent_section_of_circular_tubing(temperature=293.15,  # Kelvin
                                              pressure=101.3,  # kPa
                                              particle_diameter=3.5,  # µm
                                              particle_density=1000,  # kg/m^3
                                              tube_air_flow_rate=3,  # cc/s
                                              tube_air_velocity=False,  # m/s
                                              tube_diameter=0.0025,  # m
                                              angle_of_bend=90,  # degrees
                                              flow_type='laminar',
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
        tube_air_velocity = flow_rate2flow_velocity(tube_air_flow_rate, tube_diameter, verbose=verbose)

    if flow_type == 'auto':
        fow_type = test_flow_type(tube_diameter, tube_air_velocity, temperature, pressure, verbose=verbose)

    if flow_type == 'laminar':
        velocity_ratio = 1
        stnum = stokes_number(particle_density, particle_diameter, pressure, temperature, tube_air_velocity,
                              velocity_ratio, tube_diameter, verbose=verbose)
        fract = 1 - stnum * angle_of_bend * np.pi / 180.
    else:
        raise TypeError('sorry not implemented yet. Flow_type has to be "laminar"')
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
                                        flow_type='laminar',
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
        mean_flow_velocity = flow_rate2flow_velocity(flow_rate, tube_diameter, verbose=verbose)

    if flow_type == 'laminar':
        sv = settling_velocity(temperature, particle_density, particle_diameter, pressure, verbose=verbose)
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

    else:
        print('sorry not implemented.')
    if verbose:
        print('k770: %s' % k770)
        print('k771: %s' % k771)
        print('fraction penetrating: %s' % fract)
    return fract


def gravitational_loss_in_an_inlet(temperature=293.15,  # K
                                   pressure=101.3,  # hPa
                                   particle_diameter=15,  # um
                                   particle_density=1000,  # kg/m^3
                                   inlet_diameter=0.0127,  # m
                                   inlet_length=1.,  # m
                                   sampling_angle=45,  # deg; keep between 0 to 90°
                                   air_velocity_in_inlet=30,  # m/s;
                                   velocity_ratio=1.5,
                                   # R is 1 for isokinetic, > 1 for subisokinetic, < 1 for superisokinetic
                                   verbose=False):
    """Gravitational losses in an inlet (B&W 8-23, 8-24; W&B 6-23, 6-24)
    Not to be mixed up with gravitational loss in a circular tube

    Arguments
    ---------
    temperature:    float.
            Temperature in K.
    pressure:       float.
            Pressure in kPa.
    particle_diameter:  float.
            Aerosol particle diameter in micro meter.
    particle_density:   float.
            Density of the particle material in kg/m^3.
    inlet_diameter:     float.
            Inlet diameter in m.
    inlet_length:       float.
            Inlent length in m.
    sampling_angle:     float.
            Angle of the inlet in deg. 0 is horizontal; keep between 0 to 90°.
    air_velocity_in_inlet:  float.
            Velocity of the air in inlet in m/s.
    velocity_ratio:         float.
            Ratio between velocity outside and inside the inlet. R is 1 for isokinetic, > 1 for subisokinetic, < 1 for superisokinetic
    verbose: bool.
        if results are printed.
    """

    out = np.exp(-4.7 * K(sampling_angle, inlet_length, air_velocity_in_inlet, inlet_diameter, particle_density,
                          particle_diameter, pressure, temperature, velocity_ratio, verbose=verbose) ** 0.75)
    if verbose:
        print('Fraction lost due to gravitation: %s' % out)
    return out
