

# Sphereoids
def volume_equivalent_radius_spheroid(equatorial_axis, polar_axis):
    a = equatorial_axis
    c = polar_axis
    return  (a**2 * c)**(1/3)

def projection_area_equivalent_radius_spheroid(axis1, axis2):
    a = axis1
    c = axis2
    return (a * c) ** (1 / 2)