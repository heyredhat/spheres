"""
Functions related to Lorentz transformations/Mobius transformations.
"""
import numpy as np
import qutip as qt 
import sympy 

from .coordinates import *

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
