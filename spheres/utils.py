"""
Utility Functions
-----------------
"""

import numpy as np
import qutip as qt

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
    A, B = np.around(A), np.around(B)
    return np.array([a in B for a in A]).all()

def compare_nophase(a, b):
    """
    Compares two vectors disregarding their overall complex phase.
    """
    return np.allclose(normalize_phase(a), normalize_phase(b))
