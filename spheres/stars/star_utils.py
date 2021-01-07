"""
Useful Majorana star related functions.
"""

import quadpy
import numpy as np
import qutip as qt

from .pure import *

def spin_coherent(j, coord, from_cartesian=True,\
                            from_spherical=False,\
                            from_complex=False,\
                            from_spinor=False):
    """
    Returns the spin-j coherent state which is defined by having
    all its Majorana stars located at a single point on the sphere. 
    This point can be given in terms of cartesian, spherical, extended complex, and spinorial coordinates.

    Parameters
    ----------
        j : int
            j value which indexes the :math:`SU(2)` representation.
        coord : nd.array or qt.Qobj or complex/inf
            Coordinates specifying the direction of the spin coherent state.
        from_cartesian : bool, optional
            Whether the provided coordinates are cartesian (default).
        from_spherical : bool, optional
            Whether the provided coordinates are spherical.
        from_complex : bool, optional
            Whether the provided coordinates are extended complex.
        from_spinor : bool, optional
            Whether the provided coordinates are spinorial.

    Returns
    -------
        spin_coherent : qt.Qobj
            Spin-j coherent state in the specified direction.
    """
    if from_spherical:
        theta, phi = coord
    elif from_complex:
        theta, phi = c_sph(coord)
    elif from_spinor:
        theta, phi = c_sph(spinor_c(coord))
    elif from_cartesian:
        theta, phi = xyz_sph(coord)
    return qt.spin_coherent(j, theta, phi)

def antipodal(to_invert, from_cartesian=False,\
                         from_spherical=False):
    """
    If given an extended complex coordinate, takes the point to its antipode on the sphere via the map:
    
    .. math::

        z \\rightarrow -\\frac{z}{|z|^2} = - \\frac{1}{z^{*}}

    If :math:`z=\\infty`, :math:`z \\rightarrow 0` and if :math:`z=0`, :math:`z \\rightarrow \\infty`.

    If given a spin state or polynomial coefficients, inverts the whole sphere.
    This could be done by inverting the individual roots, or directly on the state/polynomial by
    reversing the components, complex conjugating, and multiplying every other
    component by :math:`-1`.

    If `from_cartesian=True` or `from_spherical=True`, the argument is interpreted
    as a single coordinate in terms of those coordinate systems and the flipped coordinate 
    is returned in the same.

    Parameters
    ----------
        to_invert : complex/inf or qt.Qobj or np.ndarray
            Extended complex coordinate, cartesian coordinate,
            spherical coordinate, or spin state/polynomial to invert.
    Returns
    -------
        inverted : complex/inf or qt.Qobj or np.ndarray
            Inverted extended complex coordinate, cartesian coordinate,
            spherical coordinate or spin state/polynomial.
    """
    if np.isscalar(to_invert):
        if np.isclose(to_invert, 0):
            return np.inf
        if to_invert == np.inf:
            return 0
        return -to_invert/np.abs(to_invert)**2
    if from_cartesian:
        return -1*to_invert
    if from_spherical:
        return c_sph(antipodal(sph_c(to_invert)))
    inverted = np.array([c*(-1)**(i) for i, c in enumerate(components(to_invert)[::-1].conj())])
    return qt.Qobj(inverted) if type(to_invert) == qt.Qobj else inverted

def poleflip(to_flip, from_cartesian=False,\
                      from_spherical=False):
    """
    Flips the pole of projection.  If given an extended complex coordinate, this amounts to
    projecting to the sphere via a South Pole projection and then projecting back to the plane 
    via a North Pole projection.

    More simply:

    .. math::

        z \\rightarrow \\frac{z}{|z|^2} = \\frac{1}{z^{*}}

    If :math:`z=\\infty`, :math:`z \\rightarrow 0` and if :math:`z=0`, :math:`z \\rightarrow \\infty`.

    If given a spin state or polynomial coefficients, flips the pole of projection for the entire state.
    This could be done by flipping the individual roots, or directly on the state/polynomial by
    reversing the components and complex conjugating.

    This is useful for evaluating the Majorana polynomial at :math:`\\infty`. We actually
    need a second coordinate chart. We flip the projection pole and evaluate at :math:`0` instead.

    If `from_cartesian=True` or `from_spherical=True`, the argument is interpreted
    as a single coordinate in terms of those coordinate systems and the flipped coordinate 
    is returned in the same.

    Parameters
    ----------
        to_flip : (complex/inf) or qt.Qobj or np.ndarray
            Extended complex coordinate, cartesian coordinate,
            spherical coordinate, or spin state/polynomial to flip.
        from_cartesian : bool, optional
            Whether to interpret argument as cartesian coordinates.
        from_spherical : bool, optional
            Whether to interpret argument as spherical coordinates.

    Returns
    -------
        flipped : (complex/inf) or qt.Qobj or np.ndarray
            Flipped extended complex coordinate, cartesian coordinate,
            spherical coordinate or spin state/polynomial.
    """
    if np.isscalar(to_flip):
        if np.isclose(to_flip, 0):
            return np.inf
        if to_flip == np.inf:
            return 0
        return to_flip/np.abs(to_flip)**2
    if from_cartesian:
        x, y, z = to_flip
        return np.array([x, y, -z])
    if from_spherical:
        return c_sph(poleflip(sph_c(to_flip)))
    flipped = components(to_flip)[::-1].conj()
    return qt.Qobj(flipped) if type(to_flip) == qt.Qobj else flipped

def spherical_inner(a, b):
    """
    :math:`\\langle a \\mid b \\rangle` via an integral over the sphere. 

    Parameters
    ----------
        a : func
            Normalized Majorana function.
        
        b : func
            Normalized Majorana function
    
    Returns
    -------
        inner_product : complex
    """
    scheme = quadpy.u3.get_good_scheme(19)
    return scheme.integrate_spherical(lambda sph: a(sph).conj()*b(sph))

def pauli_eigenstate(j, m, direction):
    """
    Returns eigenstates of Pauli operators.

    Parameters
    ----------
        j : float
            j value of representation.
        m : float
            m value of representation.
        direction : str
            "x", "y", or "z".
    """
    if direction == "x":
        up = np.array([1,0,0])
        down = np.array([-1,0,0])
    elif direction == "y":
        up = np.array([0,1,0])
        down = np.array([0,-1,0])
    elif direction == "z":
        up = np.array([0,0,1])
        down = np.array([0,0,-1])
    nup, ndown = [(int(2*j-i), i)\
                    for i in range(int(2*j+1))]\
                        [list(np.arange(j, -j-1, -1)).index(m)]
    return xyz_spin([up]*nup + [down]*ndown)

def basis(d, i, up='z'):
    """
    Similar to `pauli_eigenstate`, only parameterized by dimension.

    Parameters
    ----------
        d : int
            Dimension.
        
        i : int
            Basis state.
        
        up : str
            "x", "y", or "z".
    """
    if d == 0:
        return qt.identity(1)
    j = (d-1)/2
    m = np.arange(j, -j-1, -1)[i]
    return pauli_eigenstate(j, m, up)
