"""
Mixed Majorana Stars
--------------------

Implementation of the "Majorana stars" formalism for mixed states (and operators) of higher spin.

+-----------------------------------------------+------------------------------------------------+
| :py:meth:`spherical_tensor`                   | Constructs spherical tensor operator.          |
| :py:meth:`spherical_tensor_basis`             | Constructs spherical tensor basis.             |
+-----------------------------------------------+------------------------------------------------+
| :py:meth:`operator_spherical_decomposition`   | Spherical tensor decomposition.                |
| :py:meth:`spherical_decomposition_operator`   | Spherical tensor recomposition.                |
+-----------------------------------------------+------------------------------------------------+
| :py:meth:`spherical_decomposition_spins`      | Interpret decomposition as collection of spins,|
| :py:meth:`spins_spherical_decomposition`      | and back.                                      |
+-----------------------------------------------+------------------------------------------------+
| :py:meth:`operator_spins`                     | Operator to spins,                             |
| :py:meth:`spins_operator`                     | and back.                                      |
+-----------------------------------------------+------------------------------------------------+

"""

import numpy as np
import qutip as qt
from spheres.utils import *

def spherical_tensor(j, sigma, mu):
    """
    Constructs spherical tensor operator for a given :math:`j, \\sigma, \\mu`.
    """
    terms = []
    for m1 in np.arange(-j, j+1):
        for m2 in np.arange(-j, j+1):
            terms.append(\
                ((-1)**(j-m2))*\
                qt.clebsch(j, j, sigma, m1, -m2, mu)*\
                qt.spin_state(j, m1)*qt.spin_state(j, m2).dag())
    return sum(terms)

def spherical_tensor_basis(j):
    """
    Constructs a basis set of spherical tensor operators for a given :math:`j`, for all
    :math:`\\sigma` from :math:`0` to :math:`2j`, and :math:`\\mu` from :math:`-\\sigma` to :math:`\\sigma`.
    """
    T_basis = {}
    for sigma in np.arange(0, int(2*j+1)):
        for mu in np.arange(-sigma, sigma+1):
            T_basis[(sigma, mu)] = spherical_tensor(j, sigma, mu)
    return T_basis

def operator_spherical_decomposition(O, T_basis=None):
    """
    Decomposes an operator into a linear combination of spherical tensors. Constructs the latter if not supplied.
    Returns the coefficients as a dictionary for each :math:`\\sigma, \\mu`.
    """
    j = (O.shape[0]-1)/2
    if not T_basis:
        T_basis = spherical_tensor_basis(j)
    decomposition = {}
    for sigma in np.arange(0, int(2*j+1)):
        for mu in np.arange(-sigma, sigma+1):
            decomposition[(sigma, mu)] = (O*T_basis[(sigma, mu)].dag()).tr()
    return decomposition

def spherical_decomposition_operator(decomposition, T_basis=None):
    """
    Recomposes an operator from its spherical tensor decomposition. Constructs the latter if not supplied.
    """
    j = max([k[0] for k in decomposition.keys()])/2
    if not T_basis:
        T_basis = spherical_tensor_basis(j)
    terms = []
    for sigma in np.arange(0, int(2*j+1)):
        for mu in np.arange(-sigma, sigma+1):
            terms.append(decomposition[(sigma, mu)]*T_basis[(sigma, mu)])
    return sum(terms)

def spherical_decomposition_spins(decomposition):
    """
    Expresses the spherical tensor decomposition of an operator as a list of unnormalized, integer :math:`j` spin states.
    """
    max_j = max([k[0] for k in decomposition.keys()])
    return [qt.Qobj(np.array([decomposition[(j, m)]\
                        for m in np.arange(j, -j-1, -1)]))\
                            for j in np.arange(0, max_j+1)]

def spins_spherical_decomposition(spins):
    """
    Converts a list of spin states back into a dictionary of spherical tensor coefficients.
    """
    max_j = (spins[-1].shape[0]-1)/2
    decomposition = {}
    for j in np.arange(0, max_j+1):
        for m in np.arange(j, -j-1, -1):
            decomposition[(j, m)] = components(spins[int(j)])[int(j-m)]
    return decomposition

def operator_spins(O, T_basis=None):
    """
    Expresses an operator as a set of spins. Constructs the spherical tensor basis if not provided.
    This is a generalization of the Majorana representation: for an operator, instead of one constellation,
    we have several constellations on concentric spheres, whose radii can be interpreted as the norms of the
    spin states. They transform nicely under rotations and partial traces. Hermitian operators have constellations
    with antipodal symmetry, which is broken by unitary operators.
    """
    return spherical_decomposition_spins(operator_spherical_decomposition(O, T_basis=T_basis))

def spins_operator(spins, T_basis=None):
    """
    Recomposes an operator, given a list of spin states. Constructs the spherical tensor basis if not provided.
    """
    return spherical_decomposition_operator(spins_spherical_decomposition(spins), T_basis=T_basis)
