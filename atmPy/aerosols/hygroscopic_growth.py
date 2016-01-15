import numpy as np

def kappa_simple(k,RH, n = None, inverse = False):
    """Returns the growth factor as a function of kappa and RH.
    This function is based on the simplified model introduced by Rissler et al. (2006).
    It ignores the curvature effect and is therefore independend of the exact particle diameter.
    Due to this simplification this function is valid only for particles larger than 100 nm
    Patters and Kreidenweis (2007).

    Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter representation of hygroscopic
    growth and cloud condensation nucleus activity, 1961–1971. doi:10.5194/acp-7-1961-2007

    Rissler, J., Vestin, A., Swietlicki, E., Fisch, G., Zhou, J., Artaxo, P., & Andreae, M. O. (2005).
    Size distribution and hygroscopic properties of aerosol particles from dry-season biomass burning
    in Amazonia. Atmospheric Chemistry and Physics Discussions, 5(5), 8149–8207. doi:10.5194/acpd-5-8149-2005

    latex expression: $\sqrt[3]{1 + \kappa \cdot \frac{RH}{100 - RH}}$

    Arguments
    ---------
    k: float
        kappa value between 0 (no hygroscopicity) and 1.4 (NaCl; most hygroscopic
        material found in atmospheric
        aerosols)
        If inverse is True than this is the growth factor.
    RH: float
        Relative humidity -> between 0 and 100 (i don't think this simplified
        version will be correct at higher RH?!?)
    inverse: bool.
        if function is inversed, which means instead of the growth factor is
        calculated as a function of the kappa value, it is the other way around.
    Returns
    -------
    float: The growth factor of the particle.
    if n is given a further float is returned which gives the new refractiv index of the grown particle
    """

    if not inverse:
        if np.any(k > 1.4) or np.any(k < 0):
            txt = '''The kappa value has to be between 0 and 1.4.'''
            raise ValueError(txt)

    if np.any(RH > 100) or np.any(RH < 0):
        txt = """RH has to be between 0 and 100"""
        raise ValueError(txt)

    if inverse:
        kappa = lambda gf,RH: (gf**3 - 1) / (RH / (100 - RH))
        out = kappa(k,RH)
    else:
        growths_factor = lambda k,RH: (1 + (k * (RH/(100 - RH))))**(1/3.)
        out = growths_factor(k,RH)

    # adjust index of refraction
    if not inverse:
        if n:
            nw = 1.33
            n_mix = lambda n,gf: (n + ((gf-1)*nw))/gf
            return out, n_mix(n,out)
    return out