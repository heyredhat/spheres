"""
Majorana Stars
--------------

Implementation of the "Majorana stars" formalism for higher spin.

+-----------------------+-------------------------------+
| :py:meth:`c_xyz`      | Extended complex to cartesian |
| :py:meth:`xyz_c`      | and back.                     |
+-----------------------+-------------------------------+
| :py:meth:`xyz_sph`    | Cartesian to spherical        |
| :py:meth:`sph_xyz`    | and back.                     |
+-----------------------+-------------------------------+
| :py:meth:`c_sph`      | Extended complex to spherical |
| :py:meth:`sph_c`      | and back.                     |
+-----------------------+-------------------------------+
| :py:meth:`c_spinor`   | Extended complex to spinor    |
| :py:meth:`spinor_c`   | and back.                     |
+-----------------------+-------------------------------+
| :py:meth:`xyz_spinor` | Cartesian to spinor           |
| :py:meth:`spinor_xyz` | and back.                     |
+-----------------------+-------------------------------+
| :py:meth:`spinor_sph` | Spinor to spherical           |
| :py:meth:`sph_spinor` | and back.                     |
+-----------------------+-------------------------------+

+-------------------------+------------------------+
| :py:meth:`spin_poly`    | Spin to polynomial     |
| :py:meth:`poly_spin`    | and back.              |
+-------------------------+------------------------+
| :py:meth:`poly_roots`   | Polynomial to extended |
| :py:meth:`roots_poly`   | complex roots and back.|
+-------------------------+------------------------+
| :py:meth:`spin_xyz`     | Spin to cartesian roots|
| :py:meth:`xyz_spin`     | and back.              |
+-------------------------+------------------------+
| :py:meth:`spin_spinors` | Spin to spinorial roots|
| :py:meth:`spinors_spin` | and back.              |
+-------------------------+------------------------+
| :py:meth:`spin_c`       | Spin to extended       |
| :py:meth:`c_spin`       | complex roots and back.|
+-------------------------+------------------------+
| :py:meth:`spin_sph`     | Spin to spherical roots|
| :py:meth:`sph_spin`     | and back.              |
+-------------------------+------------------------+

+--------------------------+------------------------+
| :py:meth:`spin_coherent` | Spin coherent state.   |
+--------------------------+------------------------+
| :py:meth:`antipodal`     | Invert coordinates     |
|                          | on sphere.             |
+--------------------------+------------------------+
| :py:meth:`poleflip`      | Flip stereographic     |
|                          | projection pole.       |
+--------------------------+------------------------+
| :py:meth:`mobius`        | Construct Möbius       |
|                          | transformation.        |
+--------------------------+------------------------+
|:py:meth:`xyz_eigenstates`| Construct XYZ          |
|                          | eigenstates.           |
+--------------------------+------------------------+

"""

import numpy as np
import qutip as qt

from itertools import combinations

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

def c_sph(c):
    """
    Converts extended complex coordinate to spherical coordinates.

    Parameters
    ----------
    c : complex or inf
        Extended complex coordinate.

    Returns
    -------
    sph : np.ndarray
        Spherical coordinates :math:`r, \\phi, \\theta`.
    """
    return xyz_sph(c_xyz(c))

def sph_c(sph):
    """
    Converts spherical coordinates to extended complex coordinate.

    Parameters
    ----------
    sph : np.ndarray
        Spherical coordinates :math:`r, \\phi, \\theta`.

    Returns
    -------
    c : complex or inf
        Extended complex coordinate.
    """
    return xyz_c(sph_xyz(sph))

def c_spinor(c):
    """
    Converts extended complex coordinate to a spinor.

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
    Converts cartesian coordinates to a spinor by reverse stereographic projection
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

def spinor_sph(spinor):
    """
    Converts spinor to spherical coordinates. 

    Parameters
    ----------
    spinor : qt.Qobj
        Spinor.

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
    sph : np.ndarray
        Spherical coordinates :math:`r, \\phi, \\theta`.

    Returns
    -------
    spinor : qt.Qobj
        Spinor.
    """
    return xyz_spinor(sph_xyz(sph))

def spin_poly(spin, projective=False,\
                    homogeneous=False,\
                    cartesian=False,\
                    spherical=False,\
                    normalized=False):
    """
    Converts a spin into its Majorana polynomial, which is defined as follows:

    .. math::

        p(z) = \\sum_{m=-j}^{m=j} (-1)^{j+m} \\sqrt{\\frac{(2j)!}{(j-m)!(j+m)!}} a_{j+m} z^{j-m}

    Here, the :math:`a`'s run through the components of the spin in the :math:`\\mid j, m\\rangle` representation. 
    Note that :math:`\\frac{(2j)!}{(j-m)!(j+m)!}` amounts to: :math:`\\binom{2j}{j+m}`, a binomial coefficient.

    By default, returns the coefficients of the Majorana polynomial as an np.ndarray.

    If `projective=True`, returns a function which takes an extended complex coordinate
    as an argument, and which evaluates the polynomial at that point. Note that to evaluate the polynomial
    at :math:`\\infty`, we flip the stereographic projection axis and evaluate the latter polynomial at 0
    (and then complex conjugate). Insofar as pole flipping causes the highest degree term to become the lowest degree/constant term,
    evaluating at :math:`\\infty` amounts to returning the first coefficient.

    If `homogeneous=True`, returns a function which takes a spinor (as an nd.array, qt.Qobj, or two separate complex coordinates)
    and evaluates the homogeneous Majorana polynomial:

     .. math::

        p(z, w) = \\sum_{m=-j}^{m=j} (-1)^{j+m} \\sqrt{\\frac{(2j)!}{(j-m)!(j+m)!}} a_{j+m} w^{j-m} z^{j+m}

    (N.b. currently the normalization is off for the homogeneous case, but the correct roots are obtained.)

    If `cartesian=True`, returns a function with takes cartesian coordinates (as an nd.array or three sepaparate real components)
    and evaluates the Majorana polynomial by first converting the cartesian coordinates to an extended complex coordinate.

    If `spherical=True`, returns a function with takes spherical coordinates (as an nd.array or three sepaparate real components)
    and evaluates the Majorana polynomial by first converting the spherical coordinates to an extended complex coordinate.

    If `normalized=True`, returns the normalized versions of any of the above functions. Note that the normalized
    versions are no longer analytic/holomorphic. Given a extended complex coordinate :math:`z = re^{i\\theta}` (or :math:`\\infty`),
    the normalization factor is:

    .. math::

        \\frac{e^{-2ij\\theta}}{(1+r^2)^j}

    If :math:`z=\\infty`, again since we flip the poles, we use :math:`z=0`.
    When normalized, the resulting functions are equivalent to:

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
    if projective or cartesian or spherical:
        def __poly__(z):
            normalization = np.exp(-2j*j*np.angle(z))/(1+abs(z if z != np.inf else 0)**2)**j if normalized else 1
            return normalization*(poly[0] if z == np.inf else \
                        sum([poly[int(j+m)]*z**(j-m) for m in np.arange(-j, j+1)]))
        if projective:
            return __poly__
        if cartesian:
            def __cartesian__(*args):
                xyz = args[0] if len(args) == 1 else np.array(args)
                return __poly__(xyz_c(xyz))
            return __cartesian__
        if spherical:
            def __spherical__(*args):
                sph = args[0] if len(args) == 1 else np.array(args)
                return __poly__(sph_c(sph))
            return __spherical__
    if homogeneous: 
        def __hompoly__(*args):
            z, w = components(args[0]) if len(args) == 1 else args
            return sum([poly[int(j+m)]*(w**(j-m))*(z**(j+m))\
                        for m in np.arange(-j, j+1)])
        return __hompoly__
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

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

    Returns
    -------
        xyz : np.ndarray
            A array with shape (2j, 3) containing the 2j cartesian coordinates of the stars.
    """
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
    return poly_spin(roots_poly([xyz_c(star) for star in xyz]))

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
    if from_cartesian:
        r, phi, theta = xyz_sph(coord)
    if from_spherical:
        r, phi, theta = coord
    if from_complex:
        r, phi, theta = c_sph(coord)
    if from_spinor:
        r, phi, theta = c_sph(spinor_c(coord))
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

def mobius(abcd):
    """
    Given parameters :math:`\\begin{pmatrix} a & b \\\\ b & c \\end{pmatrix}`,
    arranged in a 2x2 matrix, returns a function which implements the 
    corresponding `Möbius transformation <https://en.wikipedia.org/wiki/M%C3%B6bius_transformation>`_

    .. math::

        f(z) = \\frac{az+b}{cz+d}

    which acts on the extended complex plane. Note that if :math:`c \\neq 0`, we have:

    .. math::

        f(-\\frac{d}{c}) = \\infty

        f(\\infty) = \\frac{a}{c}

    And if :math:`c = 0`, we have:

    .. math::

        f(\\infty) = \\infty

    Parameters
    ----------
        abcd : np.ndarray or qt.Qobj
            2x2 matrix representing Möbius parameters.

    Returns
    -------
        mobius : func
            A function which takes an extended complex coordinate as input, 
            and returns an extended complex coordinate as output.

    Raises
    ------
        Exception
            If :math:`ad = bc`, then :math:`f(z) = \\frac{a}{c}`, a constant function,
            which doesn't qualify as a Möbius transformation.
    """
    if type(abcd) == qt.Qobj:
        abcd = abcd.full()
    a, b, c, d = abcd.reshape(4)
    if a*d - b*c == 0:
        raise Exception("Möbius transformation doesn't satisfy ad-bc != 0.")
    def __mobius__(z):
        if c != 0:
            if z == -d/c:
                return np.inf
            if z == np.inf:
                return a/c
        else:
            if z == np.inf:
                return np.inf
        return (a*z + b)/(c*z + d)
    return __mobius__

def xyz_eigenstates(j, m, direction):
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

