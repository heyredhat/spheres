"""
Utility Functions
--------------

"""

import numpy as np
import qutip as qt

def normalize(v):
    """
    Normalizes numpy vector. Simply passes the vector through
    if its norm is 0.
    """
    n = np.linalg.norm(v)
    return v/n if not np.isclose(n, 0) else v

def components(q):
    """
    Extracts components of qt.Qobj, whether bra or ket.
    """
    return q.full().T[0] if q.type == "ket" else q.full()[0]

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

def rand_c():
    """
    Generates random complex number whose real and imaginary parts
    are normally distributed.  
    """
    return np.random.randn() + 1j*np.random.randn()