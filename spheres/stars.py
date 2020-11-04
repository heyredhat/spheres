"""
Majorana Stars
--------------

Implementation of the "Majorana stars" formalism for higher spin.

"""

import numpy as np
import qutip as qt

from spheres.utils import *

def c_xyz(c, pole="south"):
    """
    `Stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`_ 
    from the extended complex plane to the unit sphere. Given coordinate :math:`c=x+iy` or :math:`\\infty`:

        | If :math:`c = \\infty`, returns :math:`(0,0,-1)`.
        | Otherwise, returns :math:`(\\frac{2x}{1+x^2+y^2}, \\frac{2y}{1+x^2+y^2}, \\frac{1-x^2-y^2}{1+x^2+y^2})`.

    Parameters
    ----------
    c : complex or inf
        Point on the extended complex plane.
    pole : str, default 'south'
        Whether to project from the North or South pole.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of point on unit sphere.

    """
    if(pole == "south"):
        if c == np.inf:
            return np.array([0,0,-1])
        else:
            x, y = c.real, c.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (1-x**2-y**2)/(1 + x**2 + y**2)])
    elif (pole == "north"):
        if c == np.inf:
            return np.array([0,0,1])
        else:
            x, y = c.real, c.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (-1+x**2+y**2)/(1 + x**2 + y**2)])

def xyz_c(xyz, pole="south"):
    """
    Reverse `Stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`_ 
    from the unit sphere to the extended complex plane.

        | Given :math:`(0,0,-1)`, returns :math:`\\infty`.
        | Otherwise returns :math:`c = (\\frac{x}{1+z}) + i(\\frac{y}{1+z})`.

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates of point on unit sphere.
    pole : str, default 'south'
        Whether to reverse project from the North or South pole.

    Returns
    -------
    complex or inf
        Extended complex coordinate.

    """
    x, y, z = xyz
    if pole == "south":
        if np.isclose(z, -1):
            return np.inf
        else:
            return x/(1+z) + 1j*y/(1+z)
    elif pole == "north":
        if np.isclose(z, 1):
            return np.inf
        else:
            return x/(1-z) + 1j*y/(1-z)

def xyz_sph(xyz):
    """
    Converts cartesian coordinates :math:`(x, y, z)` to spherical coordinates :math:`(r, \\phi, \\theta)`.
    We use the physicist's convention: 

        | radius :math:`r = \\sqrt{x^2 + y^2 + z^2} \\in [0, \\infty)`
        | azimuth :math:`\\phi = \\arctan{\\frac{y}{x}} \\in [0, 2\\pi]`
        | inclination: :math:`\\theta = \\arccos{\\frac{z}{\\sqrt{x^2 + y^2 + z^2}}} \\in [0, \\pi]`

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates.

    Returns
    -------
    sph : np.ndarray
        Spherical coordinates.
    """
    x, y, z = xyz
    return np.array([np.sqrt(x**2 + y**2 + z**2),\
                     np.mod(np.arctan2(y, x), 2*np.pi),\
                     np.arccos(z/np.sqrt(x**2 + y**2 + z**2))])

def sph_xyz(sph):
    """
    Converts spherical coordinates :math:`(r, \\phi, \\theta)` to cartesian coordinates :math:`(x, y, z)`.
    We use the physicist's convention: :math:`r \\in [0, \\infty)`, :math:`\\phi \\in [0, 2\\pi]`, :math:`\\theta \\in [0, \\pi]`.
    
        | :math:`x = r\\sin{\\theta}\\cos(\\phi)`
        | :math:`y = r\\sin{\\theta}\\sin(\\phi)`
        | :math:`z = r\\cos(\\theta)`

    Parameters
    ----------
    sph : np.ndarray
        Spherical coordinates.

    Returns
    -------
    xyz : np.ndarray
        Cartesian coordinates.
    """
    r, phi, theta = sph
    return np.array([r*np.sin(theta)*np.cos(phi),\
                     r*np.sin(theta)*np.sin(phi),\
                     r*np.cos(theta)])

def c_spinor(c):
    """
    Converts extended complex coordinate to spinor.

        | If :math:`c = \\infty`, returns :math:`\\begin{pmatrix} 0 \\\\ 1 \\end{pmatrix}`.
        | Otherwise, returns :math:`\\frac{1}{\\sqrt{1+|c|^2}} \\begin{pmatrix} 1 \\\\ c \\end{pmatrix}`
    
    Parameters
    ----------
    c : complex or inf
        Extended complex coordinate.

    Returns
    -------
    spinor : qt.Qobj
        Normalized spinor.
    """
    return qt.Qobj(np.array([0, 1])) if c == np.inf \
        else (1/np.sqrt(1+np.abs(c)**2))*qt.Qobj(np.array([1, c]))

def spinor_c(spinor):
    """
    Converts spinor :math:`\\begin{pmatrix} a \\\\ b \\end{pmatrix}` to extended complex coordinate.

        | If :math:`a = 0`, returns :math:`\\infty`. 
        | Otherwise returns :math:`\\frac{b}{a}`.

    Parameters
    ----------
    spinor : qt.Qobj
        Normalized spinor.

    Returns
    -------
    c : complex or inf
        Extended complex coordinate.
    """
    a, b = components(spinor)
    return b/a if not np.isclose(a, 0) else np.inf

def xyz_spinor(xyz):
    """
    Converts cartesian coordinates to spinor by reverse stereographic projection
    to the extended complex plane, and then lifting the latter to a 2-vector. 

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates.

    Returns
    -------
    spinor : qt.Qobj
        Spinor.
    """
    return c_spinor(xyz_c(xyz))

def spinor_xyz(spinor):
    """
    Converts spinor :math:`\\mid \\psi \\rangle` to cartesian coordinates by taking the expectation values
    with the three Pauli matrices: :math:`(\\langle \\psi \\mid X \\mid \\psi \\rangle, \\langle \\psi \\mid Y \\mid \\psi \\rangle, \\langle \\psi \\mid Z \\mid \\psi \\rangle)`.

    Parameters
    ----------
    spinor : qt.Qobj
        Spinor.

    Returns
    -------
    xyz : np.ndarray
        Cartesian coordinates.
    """
    return np.array([qt.expect(qt.sigmax(), spinor),\
                     qt.expect(qt.sigmay(), spinor),\
                     qt.expect(qt.sigmaz(), spinor)])
