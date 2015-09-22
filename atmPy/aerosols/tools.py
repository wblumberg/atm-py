# all functions are based on http://aerosols.wustl.edu/AAARworkshop08/software/AEROCALC-11-3-03.xls

import numpy as np
import warnings


def test_flow_type_in_tube(tube_diameter, tube_air_velocity, temperature, pressure, verbose=False):
    rey_num = flow_reynolds_number(tube_diameter, tube_air_velocity, temperature, pressure, verbose=verbose)
    if np.all(rey_num < 2000):
        flowtype = 'laminar'
    elif np.all(rey_num > 4000):
        flowtype = 'turbulent'
    else:
        txt = """Flowtype can not be detected. Flow type is ambigues."""
        raise ValueError(txt)

    if verbose:
        txt = """Flow type: %s""" % flowtype
        print(txt)

    return flowtype


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
