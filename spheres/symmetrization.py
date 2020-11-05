"""
Symmetrization
--------------

"""

import numpy as np
import qutip as qt
from itertools import permutations, product

from spheres.utils import *

def symmetrize(pieces):
    """
    Given a list of quantum states, constructs their symmetrized
    tensor product. If given a multipartite quantum state instead, 
    we sum over all permutations on the subsystems.

    Parameters
    ----------
        pieces : list or qt.Qobj
            List of quantum states or a multipartite state.

    Returns
    -------
        sym : qt.Qobj
            Symmetrized tensor product.
    """
    if type(pieces) == list:
        return sum([qt.tensor(*[pieces[i] for i in perm])\
                for perm in permutations(range(len(pieces)))]).unit()
    else:
        return sum([pieces.permute(perm)\
                for perm in permutations(range(len(pieces.dims[0])))]).unit()
                
def spin_sym(j):
    """
    Constructs an isometric linear map from spin-j states to
    permutation symmetric states of 2j spin-:math:`\\frac{1}{2}`'s.

    Parameters
    ----------
        j : int
            j value.

    Returns
    -------
        S : qt.Qobj
            Linear map from :math:`2j+1` dimensions to :math:`2^{2j}` dimensions.

    """
    if j == 0:
        return qt.Qobj(1)
    S = qt.Qobj(np.vstack([\
                    components(symmetrize(\
                           [qt.basis(2,0)]*int(2*j-i)+\
                           [qt.basis(2,1)]*i))\
                for i in range(int(2*j+1))]).T)
    S.dims =[[2]*int(2*j), [int(2*j+1)]]
    return S
