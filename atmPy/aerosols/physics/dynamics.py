import numpy as _np


def dynamic_shape_factor4shperoids(polar2equatorial_axis_ratio, orientation2flow = 'perpendicular'):
    """
    Returns the dynamic shape factor of spheroids depending on their aspect ratio and their orientation with repect to
    the direction of the flow.
    Args:
        polar2equatorial_axis_ratio: float or ndarray
        orientation2flow: "perpendicular or parallel"

    Returns:
        float or ndarray
    """
    def k_prolate_perp(q):
        """spheroid shape: https://en.wikipedia.org/wiki/Spheroid
        function ref:eq 13 in Kasper, G., 1982. Dynamics and Measurement of Smokes. I Size Characterization of Nonspherical Particles. Aerosol Sci. Technol. 1, 187–199. doi:10.1080/02786828208958587"""
        out = (( 8 /3) * ( q **2 - 1) * q** (-1 / 3)) / (
            ((((2 * q ** 2) - 3) / _np.sqrt(q ** 2 - 1)) * _np.log(q + _np.sqrt(q ** 2 - 1))) + q)
        return out

    def k_prolate_parallel(q):
        """spheroid shape: https://en.wikipedia.org/wiki/Spheroid
        function ref:eq 14 in Kasper, G., 1982. Dynamics and Measurement of Smokes. I Size Characterization of Nonspherical Particles. Aerosol Sci. Technol. 1, 187–199. doi:10.1080/02786828208958587"""
        out = ((4 / 3) * (q ** 2 - 1) * q ** (-1 / 3)) / (
            ((((2 * q ** 2) - 1) / _np.sqrt(q ** 2 - 1)) * _np.log(q + _np.sqrt(q ** 2 - 1))) - q)
        return out

    def k_oblate_perp(q):
        """spheroid shape: https://en.wikipedia.org/wiki/Spheroid
        function ref:eq 15 in Kasper, G., 1982. Dynamics and Measurement of Smokes. I Size Characterization of Nonspherical Particles. Aerosol Sci. Technol. 1, 187–199. doi:10.1080/02786828208958587"""
        out = ((8 / 3) * (q ** 2 - 1) * q ** (-1 / 3)) / (
            ((((2 * q ** 2) - 3) / _np.sqrt(1 - q ** 2)) * _np.arccos(q)) + q)
        return out

    def k_oblate_parallel(q):
        """spheroid shape: https://en.wikipedia.org/wiki/Spheroid
        function ref:eq 16 in Kasper, G., 1982. Dynamics and Measurement of Smokes. I Size Characterization of Nonspherical Particles. Aerosol Sci. Technol. 1, 187–199. doi:10.1080/02786828208958587"""
        out = ((4 / 3) * (q ** 2 - 1) * q ** (-1 / 3)) / (
            ((((2 * q ** 2) - 1) / _np.sqrt(1 - q ** 2)) * _np.arccos(q)) - q)
        return out

    isndarray = True
    if type(polar2equatorial_axis_ratio).__name__ != 'ndarray':
        isndarray = False
        polar2equatorial_axis_ratio = _np.array([polar2equatorial_axis_ratio])

    out = _np.zeros(polar2equatorial_axis_ratio.shape)

    if orientation2flow == 'perpendicular':
        #     if polar2equatorial_axis_ratio >= 1:
        #         if orientation2flow = 'perpendicular':
        out[polar2equatorial_axis_ratio >= 1] = k_prolate_perp(
            polar2equatorial_axis_ratio[polar2equatorial_axis_ratio >= 1])
        out[polar2equatorial_axis_ratio < 1] = k_oblate_perp(
            polar2equatorial_axis_ratio[polar2equatorial_axis_ratio < 1])
    elif orientation2flow == 'parallel':
        #     if polar2equatorial_axis_ratio >= 1:
        #         if orientation2flow = 'perpendicular':
        out[polar2equatorial_axis_ratio >= 1] = k_prolate_parallel(
            polar2equatorial_axis_ratio[polar2equatorial_axis_ratio >= 1])
        out[polar2equatorial_axis_ratio < 1] = k_oblate_parallel(
            polar2equatorial_axis_ratio[polar2equatorial_axis_ratio < 1])
    else:
        raise ValueError(
            'Keyword "orientation2flow" has to be either "{}" or "{}"! (it is "{}")'.format('perpendicular', 'parallel',
                                                                                            orientation2flow))

    if isndarray:
        return out
    else:
        return out[0]