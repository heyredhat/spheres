"""
Utility Functions
-----------------
"""

import numpy as np
import qutip as qt

factorial = np.math.factorial

def rand_c():
    """
    Generates random extended complex coordinate whose real and imaginary parts
    are normally distributed, and ten percent of the time, we return :math:`\\infty`.
    """
    return np.random.randn() + 1j*np.random.randn() if np.random.random() > 0.1 else np.inf

def rand_xyz(n=1):
    """
    Generates n random points on the unit sphere in cartesian coordinates.
    """
    if n == 1:
        return normalize(np.random.randn(3))
    return np.array([normalize(np.random.randn(3)) for i in range(n)])

def rand_sph(unit=False):
    """
    Generates random spherical coordinate. If `unit=True`, the point
    lies on the surface of the sphere.
    """
    return np.array([1 if unit else np.random.random(),\
                     2*np.pi*np.random.random(),\
                     np.pi*np.random.random()])

def components(q):
    """
    Extracts components of qt.Qobj, whether bra or ket.
    """
    return (q.full().T[0] if q.type == "ket" else q.full()[0])\
                if type(q) == qt.Qobj else q

def normalize(v):
    """
    Normalizes numpy vector. Simply passes the vector through
    if its norm is 0.
    """
    n = np.linalg.norm(v)
    return v/n if not np.isclose(n, 0) else v

def phase(v):
    """
    Extracts phase of complex vector (np.ndarray or qt.Qobj) by finding
    the first non-zero component and returning its phase.
    """
    v = components(v) if type(v) == qt.Qobj else v
    return np.exp(1j*np.angle(v[(v!=0).argmax(axis=0)]))

def normalize_phase(v):
    """
    Normalizes the phase of a complex vector (np.ndarray or qt.Qobj).
    """
    return v/phase(v)

def compare_unordered(A, B, decimals=5):
    """
    Compares two sets of vectors regardless of their ordering.
    """
    A, B = np.around(A, decimals=decimals), np.around(B, decimals=decimals)
    return np.array([a in B for a in A]).all()

def compare_nophase(a, b):
    """
    Compares two vectors disregarding their overall complex phase.
    """
    return np.allclose(normalize_phase(a), normalize_phase(b))

def compare_spinors(A, B, decimals=5):
    """
    Compares two lists of spinors, disregarding both their phases,
    as well as their ordering in the list.
    """
    A = np.array([components(normalize_phase(a)) for a in A])
    B = np.array([components(normalize_phase(b)) for b in B])
    return compare_unordered(A, B, decimals=decimals)

def so3_generators():
    """
    Returns :math:`SO(3)` generators :math:`L_{x}, L_{y}, L_{z}`.
    """
    Lx = np.array([[0,0,0],\
                   [0,0,-1],\
                   [0,1,0]])
    Ly = np.array([[0,0,1],\
                   [0,0,0],\
                   [-1,0,0]])
    Lz = np.array([[0,-1,0],\
                   [1,0,0],\
                   [0,0,0]])
    return Lx, Ly, Lz

def bitstring_basis(bitstring, dims=2):
    """
    Generates a basis vector corresponding to a given bitstring, 
    which may be a list of integers or a string of integers. The
    dimensionality of each tensor factor is given by dims, which may
    be either an integer (all the same dimensionality) or a list 
    (a dimension for each factor).
    """
    if type(bitstring) == str:
        bitstring = [int(s) for s in bitstring]
    if type(dims) != list:
        dims = [dims]*len(bitstring)
    return qt.tensor(*[qt.basis(dims[i], i)\
                for i in range(len(bitstring))])

def fix_stars(old_stars, new_stars):
    """
    Try to adjust the ordering of a list of stars to keep continuity.
    """
    if np.all([np.allclose(old_stars[0], old_star) for old_star in old_stars]):
        return new_stars
    ordering = [None]*len(old_stars)
    for i, old_star in enumerate(old_stars):
        dists = np.array([np.linalg.norm(new_star-old_star) for new_star in new_stars])
        minim = np.argmin(dists)
        if np.count_nonzero(dists == dists[minim]) == 1:
            ordering[i] = new_stars[minim]
        else:
            return new_stars
    return ordering

def flatten(to_flatten):
    """
    Flatten list of lists.
    """
    return [item for sublist in to_flatten for item in sublist]

def binomial(n, k):
    """
    Returns the binomial coefficient :math:`\\binom{n}{k} = \\frac{n!}{k!(n-k)!}`
    """
    return int(factorial(n)/(factorial(k)*factorial(n-k)))
