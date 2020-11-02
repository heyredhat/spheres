"""
Implementation of the "Majorana stars" formalism.

"""

import numpy as np

def c_xyz(c, pole="south"):
    """
    Stereographic projection of complex point to the unit sphere.

    .. math:: 

        (a +b)^2

    Parameters
    ----------
    c : complex
        Point on the complex plane
    pole : str
        Whether to project from the North or South pole

    Returns
    -------
    np.ndarray
        XYZ coords
    """
    if(pole == "south"):
        if c == float("Inf"):
            return np.array([0,0,-1])
        else:
            x, y = c.real, c.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (1-x**2-y**2)/(1 + x**2 + y**2)])
    elif (pole == "north"):
        if c == float("Inf"):
            return np.array([0,0,1])
        else:
            x, y = c.real, c.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (-1+x**2+y**2)/(1 + x**2 + y**2)])

def xyz_c(xyz, pole="south"):
    """
    Reverse stereographic projection from the unit sphere to the complex plane.
    """
    x, y, z = xyz
    if (pole=="south"):
        if np.isclose(z,-1):
            return float("Inf")
        else:
            return x/(1+z) + 1j*y/(1+z)
    elif (pole=="north"):
        if np.isclose(z,1):
            return float("Inf")
        else:
            return x/(1-z) + 1j*y/(1-z)