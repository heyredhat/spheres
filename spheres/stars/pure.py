"""
Implementation of the "Majorana stars" formalism for pure states of higher spin.
"""

import numpy as np
import qutip as qt
from itertools import *

from ..utils import *
from ..coordinates import *

def spin_poly(spin, projective=False,\
                    homogeneous=False,\
                    cartesian=False,\
                    spherical=False,\
                    normalized=False,\
                    for_integration=False):
    """
    Converts a spin into its Majorana polynomial, which is defined as follows:

    .. math::

        p(z) = \\sum_{m=-j}^{m=j} (-1)^{j+m} \\sqrt{\\frac{(2j)!}{(j-m)!(j+m)!}} a_{j+m} z^{j-m}

    Here, the :math:`a`'s run through the components of the spin in the :math:`\\mid j, m\\rangle` representation. 
    Note that :math:`\\frac{(2j)!}{(j-m)!(j+m)!}` amounts to: :math:`\\binom{2j}{j+m}`, a binomial coefficient.

    By default, returns the coefficients of the Majorana polynomial as an np.ndarray.

    If `projective=True`, returns a function which takes an (array of) extended complex coordinate(s)
    as an argument, and which evaluates the polynomial at that/those point(s). Note that to evaluate the polynomial
    at :math:`\\infty`, we flip the stereographic projection axis and evaluate the latter polynomial at 0
    (and then complex conjugate). Insofar as pole flipping causes the highest degree term to become the lowest degree/constant term,
    evaluating at :math:`\\infty` amounts to returning the first coefficient. 

    If `homogeneous=True`, returns a function which takes a spinor (as an nd.array or qt.Qobj)
    and evaluates the (unnormalized) homogeneous Majorana polynomial:

     .. math::

        p(z, w) = \\sum_{m=-j}^{m=j} (-1)^{j+m} \\sqrt{\\frac{(2j)!}{(j-m)!(j+m)!}} a_{j+m} w^{j-m} z^{j+m}

    If `cartesian=True`, returns a function with takes cartesian coordinates
    and evaluates the Majorana polynomial by first converting the cartesian coordinates to extended complex coordinates.

    If `spherical=True`, returns a function with takes spherical coordinates
    and evaluates the Majorana polynomial by first converting the spherical coordinates to extended complex coordinates.

    If `normalized=True`, returns the normalized versions of any of the above functions. Note that the normalized
    versions are no longer analytic/holomorphic. Given a extended complex coordinate :math:`z = re^{i\\theta}` (or :math:`\\infty`),
    the normalization factor is:

    .. math::

        \\frac{e^{-2ij\\theta}}{(1+r^2)^j}

    If :math:`z=\\infty`, again since we flip the poles, we use :math:`z=0`.

    If `for_integration=True`, returns normalized function that takes spherical coordinates, with an extra normalization
    factor of :math:`\\sqrt{\\frac{2j+1}{4\\pi}}` so that the integral over the sphere gives a normalized amplitude. 
    Note that coordinates must be given in the form of [[:math:`\\theta`'s], [:math:`\\phi`'s]].

    When normalized, evaluating the Majorana polynomial is equivalent to evaluating:

    .. math::
        \\langle -xyz \\mid \\psi \\rangle

    Where :math:`\\mid xyz \\rangle` refers to the spin coherent state which has all its "stars" at 
    cartesian coordinates :math:`x, y, z`, and :math:`\\mid \\psi \\rangle` is the spin in the :math:`\\mid j, m\\rangle`
    representation. In other words, evaluating a normalized Majorana function at :math:`x, y, z` is equivalent to evaluating:

    .. code-block::

       spin_coherent(j, -xyz).dag()*spin

    Which is the inner product between the spin and the spin coherent state antipodal to :math:`x, y, z`
    on the sphere. Since the Majorana stars are zeros of this function, we can interpret them
    as picking out those directions for which there's 0 probability that all the angular momentum is concentrated
    in the opposite direction. Insofar as we can think of each star as contributing a quantum of angular momentum :math:`\\frac{1}{2}`
    in that direction, naturally there's no chance that *all* the angular momentum is concentrated opposite to any of those points.
    By the fundamental theorem of algebra, knowing these points is equivalent to knowing the entire quantum state.

    Parameters
    ----------
    spin : qt.Qobj
        Spin-j state.

    projective : bool, optional
        Whether to return Majorana polynomial as a function of an extended complex coordinate.

    homogeneous : bool, optional
        Whether to return Majorana polynomial as a function of a spinor.

    cartesian : bool, optional
        Whether to return Majorana polynomial as a function of cartesian coordinates on unit sphere.

    spherical : bool, optional
        Whether to return Majorana polynomial as a function of spherical coordinates.

    normalize : bool, optional
        Whether to normalize the above functions.

    for_integration : bool, optional
        Extra normalization for integration.

    Returns
    -------
    poly : np.ndarray or func
        Either 2j+1 Majorana polynomial coefficients or else one of the above functions.
    """        
    j = (spin.shape[0]-1)/2
    v = components(spin)
    poly = np.array(\
            [v[int(m+j)]*\
                (((-1)**(int(m+j)))*\
                np.sqrt(factorial(2*j)/(factorial(j-m)*factorial(j+m))))
                    for m in np.arange(-j, j+1)])
    if for_integration:
        normalized = True
    if projective or cartesian or spherical or for_integration:
        def _poly_(z):
            normalization = np.exp(-2j*j*np.angle(z))/(1+abs(z if z != np.inf else 0)**2)**j if normalized else 1
            return normalization*(poly[0] if z == np.inf else \
                        sum([poly[int(j+m)]*z**(j-m) for m in np.arange(-j, j+1)]))
        def __poly__(z):
            return np.array([_poly_(z_) for z_ in z]) if isinstance(z, Iterable) else _poly_(z)
        if projective:
            return __poly__
        if cartesian:
            return lambda xyz: __poly__(xyz_c(xyz))
        if spherical:
            return lambda sph: __poly__(sph_c(sph))
        if for_integration:
            def __for_integration__(theta_phi):
                return np.sqrt((2*j+1)/(4*pi))*__poly__(sph_c(theta_phi.T))
            return __for_integration__
    if homogeneous: 
        def _hompoly_(spinor):
            z, w = components(spinor)
            return sum([poly[int(j+m)]*(w**(j-m))*(z**(j+m)) for m in np.arange(-j, j+1)])
        return lambda spinor: np.array([_hompoly_(spinor_) for spinor_ in spinor]) if isinstance(spinor, Iterable) else _hompoly_(spinor)
    return poly

def poly_spin(poly):
    """
    Converts a Majorana polynomial into a spin-j state.

    Parameters
    ----------
        poly : np.ndarray
            2j+1 Majorana polynomial coefficients.

    Returns
    -------
        spin : qt.Qobj
            Spin-j state.
    """
    j = (len(poly)-1)/2
    return qt.Qobj(np.array(\
            [poly[int(m+j)]/\
                (((-1)**(int(m+j)))*\
                np.sqrt(factorial(2*j)/(factorial(j-m)*factorial(j+m))))
                    for m in np.arange(-j, j+1)])).unit()

def poly_roots(poly):
    """
    Takes a Majorana polynomial to its roots. We use numpy's polynomial solver. 
    The number of initial coefficients which are 0 are intepreted as the number of
    roots at :math:`\\infty`. In other words, to the extent that the degree of 
    a Majorana polynomial corresponding to a spin-j state is less than 2j+1, we add
    that many roots at :math:`\\infty`.

    Parameters
    ----------
        poly : np.ndarray
            2j+1 Majorana polynomial coefficients.

    Returns
    -------
        roots : list
            Roots on the extended complex plane.

    """
    return [np.inf]*np.flatnonzero(poly)[0] + [complex(root) for root in np.roots(poly)]

def roots_poly(roots):
    """
    Takes a set of points on the extended complex plane and forms the polynomial 
    which has these points as roots. Roots at :math:`\\infty` turn into initial
    zero coefficients.

    Parameters
    ----------
        roots : list
            Roots on the extended complex plane.

    Returns
    -------
        poly : np.ndarray
            2j+1 Majorana polynomial coefficients.

    """
    zeros = roots.count(0j)
    if zeros == len(roots):
        return np.array([1] + [0j]*len(roots))
    poles = roots.count(np.inf)
    if poles == len(roots):
        return np.array([0j]*poles + [1])
    roots = [root for root in roots if root != np.inf]
    coeffs = np.array([((-1)**(-i))*sum([np.prod(terms)\
                        for terms in combinations(roots, i)])\
                            for i in range(len(roots)+1)])
    return np.concatenate([np.zeros(poles), coeffs])

def spin_xyz(spin):
    """
    Takes a spin-j state and returns the cartesian coordinates on the unit sphere
    corresponding to its "Majorana stars." Each contributes a quantum of angular 
    momentum :math:`\\frac{1}{2}` to the overall spin.

    Note: If given a spin-0 state, returns [[0,0,0]]. If given a state with 0 norm, returns a list of 2j 0-vectors.

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

    Returns
    -------
        xyz : np.ndarray
            A array with shape (2j, 3) containing the 2j cartesian coordinates of the stars.
    """
    if spin.shape[0] == 1:
        return np.array([[0,0,0]])
    elif spin.norm() == 0:
        return np.array([[0,0,0] for i in range(spin.shape[0]-1)])
    return np.array([c_xyz(root) for root in poly_roots(spin_poly(spin))])

def xyz_spin(xyz):
    """
    Given the cartesian coordinates of a set of "Majorana stars," returns the
    corresponding spin-j state, which is defined only up to complex phase.

    Parameters
    ----------
        xyz : np.ndarray
            A array with shape (2j, 3) containing the 2j cartesian coordinates of the stars.

    Returns
    -------
        spin : qt.Qobj
            Spin-j state.
    """
    return poly_spin(roots_poly([xyz_c(star) for star in xyz])).unit()

def spin_spinors(spin):
    """
    Takes a spin-j state and returns its decomposition into 2j spinors.

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

    Returns
    -------
        spinors : list
            2j spinors.
    """
    return [c_spinor(c) for c in poly_roots(spin_poly(spin))]

def spinors_spin(spinors):
    """
    Given 2j spinors returns the corresponding spin-j state (up to phase).

    Parameters
    ----------
        spinors : list
            2j spinors.

    Returns
    -------
        spin : qt.Qobj
            Spin-j state.
    """
    return poly_spin(roots_poly([spinor_c(spinor) for spinor in spinors]))

def spin_c(spin):
    """
    Takes a spin-j state and returns its decomposition into 2j roots on the extended complex plane.

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

    Returns
    -------
        c : list
            2j extended complex roots.
    """
    return poly_roots(spin_poly(spin))

def c_spin(c):
    """
    Takes 2j roots on the extended complex plane and returns the corresponding spin-j state (up to complex phase).

    Parameters
    ----------
        c : list
            2j extended complex roots.

    Returns
    -------
        spin : qt.Qobj
            Spin-j state.
    """
    return poly_spin(roots_poly(c))

def spin_sph(spin):
    """
    Takes a spin-j state and returns its decomposition into 2j "stars" given in spherical coordinates.

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

    Returns
    -------
        sph : np.array
            A array with shape (2j, 3) containing the 2j spherical coordinates of the stars.
    """
    return np.array([c_sph(c) for c in poly_roots(spin_poly(spin))])

def sph_spin(sph):
    """
    Takes 2j "stars" given in spherical coordinates and returns the corresponding spin-j state (up to complex phase).

    Parameters
    ----------
        sph : np.array
            A array with shape (2j, 3) containing the 2j spherical coordinates of the stars.

    Returns
    -------
        spin : qt.Qobj
            Spin-j state.
    """
    return poly_spin(roots_poly([sph_c(s) for s in sph]))