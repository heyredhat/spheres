"""
Coordinate transformations, mainly sphere related.
"""

import numpy as np
from numpy import pi

import qutip as qt
from collections.abc import Iterable

from .utils import *

def __c_xyz__(c, pole="south"):
    if c == np.inf:
        return np.array([0,0,-1]) if pole == "south" else np.array([0,0,1])
    else:
        x, y = c.real, c.imag
        return np.array([2*x/(1+x**2+y**2),\
                         2*y/(1+x**2+y**2),\
                         ((1-x**2-y**2)/(1+x**2+y**2))*\
                             (1 if pole == "south" else -1)])

def c_xyz(c, pole="south"):
    """
    `Stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`_ 
    from the extended complex plane to the unit sphere. Given coordinate :math:`c=x+iy` or :math:`\\infty`:

        | If :math:`c = \\infty`, returns :math:`(0,0,-1)`.
        | Otherwise, returns :math:`(\\frac{2x}{1+x^2+y^2}, \\frac{2y}{1+x^2+y^2}, \\frac{1-x^2-y^2}{1+x^2+y^2})`.

    Parameters
    ----------
    c : complex/inf or list/np.ndarray
        Point(s) on the extended complex plane.
    pole : str, default 'south'
        Whether to project from the North or South pole.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of point(s) on unit sphere.

    """
    return np.array([__c_xyz__(c_, pole=pole) for c_ in c]) if isinstance(c, Iterable) else __c_xyz__(c, pole=pole)

def __xyz_c__(xyz, pole="south"):
    x, y, z = xyz
    if pole == "south":
        return np.inf if np.isclose(z, -1) else x/(1+z) + 1j*y/(1+z)
    elif pole == "north":
        return np.inf if np.isclose(z, 1) else x/(1-z) + 1j*y/(1-z)

def xyz_c(xyz, pole="south"):
    """
    Reverse `Stereographic projection <https://en.wikipedia.org/wiki/Stereographic_projection>`_ 
    from the unit sphere to the extended complex plane.

        | Given :math:`(0,0,-1)`, returns :math:`\\infty`.
        | Otherwise returns :math:`c = (\\frac{x}{1+z}) + i(\\frac{y}{1+z})`.

    Parameters
    ----------
    xyz : list/np.ndarray
        Cartesian coordinates of point(s) on unit sphere.
    pole : str, default 'south'
        Whether to reverse project from the North or South pole.

    Returns
    -------
    complex/inf or np.ndarray
        Extended complex coordinate(s).

    """
    xyz = np.asarray(xyz)
    return np.array([__xyz_c__(xyz_, pole=pole) for xyz_ in xyz]) if len(xyz.shape) != 1 else __xyz_c__(xyz, pole=pole)

def xyz_sph(xyz):
    """
    Converts cartesian coordinates :math:`(x, y, z)` to spherical coordinates :math:`(\\theta, \\phi)`.
    We use the physicist's convention: 

        | inclination: :math:`\\theta = \\arccos{\\frac{z}{\\sqrt{x^2 + y^2 + z^2}}} \\in [0, \\pi]`
        | azimuth :math:`\\phi = \\arctan{\\frac{y}{x}} \\in [0, 2\\pi]`

    Parameters
    ----------
    xyz : list/np.ndarray
        (List of) Cartesian coordinates.

    Returns
    -------
    sph : np.ndarray
        (List of) Spherical coordinates.
    """
    xyz = np.asarray(xyz)
    x, y, z = xyz if len(xyz.shape) == 1 else xyz.T
    return np.array([np.arccos(z/np.sqrt(x**2+y**2+z**2)),\
                     np.mod(np.arctan2(y, x), 2*np.pi)]).T

def sph_xyz(sph):
    """
    Converts spherical coordinates :math:`(\\theta, \\phi)` to cartesian coordinates :math:`(x, y, z)`.
    We use the physicist's convention: :math:`\\theta \\in [0, \\pi]`, :math:`\\phi \\in [0, 2\\pi]`.
    
        | :math:`x = \\sin{\\theta}\\cos(\\phi)`
        | :math:`y = \\sin{\\theta}\\sin(\\phi)`
        | :math:`z = \\cos(\\theta)`

    Parameters
    ----------
    sph : list/np.ndarray
        (List of) Spherical coordinates.

    Returns
    -------
    xyz : np.ndarray
        (List of) Cartesian coordinates.
    """
    sph = np.asarray(sph)
    theta, phi = sph if len(sph.shape) == 1 else sph.T
    return np.array([np.sin(theta)*np.cos(phi),\
                     np.sin(theta)*np.sin(phi),\
                     np.cos(theta)]).T

def c_sph(c):
    """
    Converts extended complex coordinate to spherical coordinates.

    Parameters
    ----------
    c : complex/inf or list/np.ndarray
        Extended complex coordinate(s).

    Returns
    -------
    sph : np.ndarray
        (List of) Spherical coordinates :math:`\\theta, \\phi`.
    """
    return xyz_sph(c_xyz(c))

def sph_c(sph):
    """
    Converts spherical coordinates to extended complex coordinate.

    Parameters
    ----------
    sph : list/np.ndarray
        (List of) Spherical coordinates :math:`\\theta, \\phi`.

    Returns
    -------
    c : complex/inf or np.ndarray
        Extended complex coordinate(s).
    """
    return xyz_c(sph_xyz(sph))

def __c_spinor__(c):
    return qt.Qobj(np.array([0, 1])) if c == np.inf else (1/np.sqrt(1+np.abs(c)**2))*qt.Qobj(np.array([1, c]))

def c_spinor(c):
    """
    Converts extended complex coordinate to a spinor.

        | If :math:`c = \\infty`, returns :math:`\\begin{pmatrix} 0 \\\\ 1 \\end{pmatrix}`.
        | Otherwise, returns :math:`\\frac{1}{\\sqrt{1+|c|^2}} \\begin{pmatrix} 1 \\\\ c \\end{pmatrix}`
    
    Parameters
    ----------
    c : complex/inf or list/np.ndarray
        Extended complex coordinate(s).

    Returns
    -------
    spinor : list or qt.Qobj
        (List of) normalized spinor(s).
    """
    return [__c_spinor__(c_) for c_ in c] if isinstance(c, Iterable) else __c_spinor__(c)

def __spinor_c__(spinor):
    a, b = components(spinor)
    return b/a if not np.isclose(a, 0) else np.inf

def spinor_c(spinor):
    """
    Converts spinor :math:`\\begin{pmatrix} a \\\\ b \\end{pmatrix}` to extended complex coordinate.

        | If :math:`a = 0`, returns :math:`\\infty`. 
        | Otherwise returns :math:`\\frac{b}{a}`.

    Parameters
    ----------
    spinor : list or qt.Qobj
        Normalized spinor(s).

    Returns
    -------
    c : complex/inf or np.ndarray
        Extended complex coordinate(s).
    """
    return [__spinor_c__(spinor_) for spinor_ in spinor] if isinstance(spinor, Iterable) else __spinor_c__(spinor)

def xyz_spinor(xyz):
    """
    Converts cartesian coordinates to a spinor by reverse stereographic projection
    to the extended complex plane, and then lifting the latter to a 2-vector. 

    Parameters
    ----------
    xyz : list/np.ndarray
        (List of) cartesian coordinates.

    Returns
    -------
    spinor : qt.Qobj or list
        Spinor(s).
    """
    return c_spinor(xyz_c(xyz))

def __spinor_xyz__(spinor):
    return np.array([qt.expect(qt.sigmax(), spinor),\
                     qt.expect(qt.sigmay(), spinor),\
                     qt.expect(qt.sigmaz(), spinor)])

def spinor_xyz(spinor):
    """
    Converts spinor :math:`\\mid \\psi \\rangle` to cartesian coordinates by taking the expectation values
    with the three Pauli matrices: :math:`(\\langle \\psi \\mid X \\mid \\psi \\rangle, \\langle \\psi \\mid Y \\mid \\psi \\rangle, \\langle \\psi \\mid Z \\mid \\psi \\rangle)`.

    Parameters
    ----------
    spinor : list or qt.Qobj
        Spinor(s).

    Returns
    -------
    xyz : np.ndarray
       (List of) cartesian coordinates.
    """
    return [__spinor_xyz__(spinor_) for spinor_ in spinor] if isinstance(spinor, Iterable) else __spinor_xyz__(spinor)

def spinor_sph(spinor):
    """
    Converts spinor to spherical coordinates. 

    Parameters
    ----------
    spinor : list or qt.Qobj
        Spinor(s).

    Returns
    -------
    sph : np.ndarray
        Spherical coordinates :math:`r, \\phi, \\theta`.
    """
    return xyz_sph(spinor_xyz(spinor))

def sph_spinor(sph):
    """
    Converts spherical coordinates to spinor. 

    Parameters
    ----------
    sph : list or np.ndarray
        Spherical coordinates :math:`r, \\phi, \\theta`.

    Returns
    -------
    spinor : list or qt.Qobj
        Spinor(s).
    """
    return xyz_spinor(sph_xyz(sph))