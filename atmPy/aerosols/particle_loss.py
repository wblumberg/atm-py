# all functions are based on http://aerosols.wustl.edu/AAARworkshop08/software/AEROCALC-11-3-03.xls

import numpy as np


def air_density(temperature, pressure, verbose=False):
    out = 1.293 * (273.15 / temperature) * (pressure / 101.3)
    if verbose:
        print('air density: %s' % out)
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
    out = (particle_density * particle_diameter ** 2 * 0.000000000001 * slip_correction_factor(pressure,
                                                                                               particle_diameter,
                                                                                               verbose=verbose) / (
           18 * air_viscosity(temperature, verbose=verbose))) * air_velocity_in_inlet * velocity_ratio / inlet_diameter
    if verbose:
        print('gravitational dep parameter: %s' % out)
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


###########################
def gravitational_loss_in_circular_tube(temperature=293.15,  # Kelvin
                                        pressure=101.3,  # kPa
                                        particle_diameter=10,  # µm
                                        particle_density=1000,  # kg/m^3
                                        tube_diameter=0.01,  # m
                                        tube_length=0.1,  # m
                                        incline_angle=60,  # degrees from horizontal (0-90)
                                        flow_rate=3,  # cc/s
                                        mean_flow_velocity=False  # 0.1061    # m/s)

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
