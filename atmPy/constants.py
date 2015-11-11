# -*- coding: utf-8 -*-
"""
Common physical constants.

@author: mtat76
"""
from math import pi

# Electron charge in Coulombs
e = 1.60218e-19

# dielectric constant
eps0 = 8.8542e-12

# Boltzmann's constant
k = 1.3807e-23

# convert angle to radians
a2r = lambda x: x/180*pi

# convert radians to angle
r2a = lambda x: x/pi*180