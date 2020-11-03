"""
Implementation of the "Majorana stars" formalism.

"""

def c_xyz(z, pole="south"):
    """
    `Stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`_ 
    from the extended complex plane to the unit sphere.

    Given coordinate :math:`z=x+iy` or :math:`\\infty`,
    returns :math:`(0,0,-1)` if :math:`z = \\infty`, otherwise:

    .. math:: 

        (\\frac{2x}{1+x^2+y^2}, \\frac{2y}{1+x^2+y^2}, \\frac{1-x^2-y^2}{1+x^2+y^2})

    Parameters
    ----------
    c : complex or inf
        Point on the extended complex plane.
    pole : str, default 'south'
        Whether to project from the North or South pole.

    Returns
    -------
    np.ndarray
        XYZ coordinates.

    """
    if(pole == "south"):
        if z == float("Inf"):
            return np.array([0,0,-1])
        else:
            x, y = z.real, z.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (1-x**2-y**2)/(1 + x**2 + y**2)])
    elif (pole == "north"):
        if z == float("Inf"):
            return np.array([0,0,1])
        else:
            x, y = z.real, z.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (-1+x**2+y**2)/(1 + x**2 + y**2)])

def xyz_c(xyz, pole="south"):
    """
    Reverse `Stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`_ 
    from the unit sphere to the extended complex plane.

    Given :math:`(0,0,-1)`, returns :math:`\\infty`. Otherwise: 

    .. math:: 

        z = (\\frac{x}{1+z}) + i(\\frac{y}{1+z})

    Parameters
    ----------
    xyz : np.ndarray
        Point on the unit sphere.
    pole : str, default 'south'
        Whether to reverse project from the North or South pole.

    Returns
    -------
    complex or inf
        Extended complex coordinate.

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