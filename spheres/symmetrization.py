"""
Symmetrization
--------------

+-------------------------+----------------------------------------+
| :py:meth:`symmetrize`        | Symmetrizes states.               |
| :py:meth:`symmetrized_basis` | Constructs symmetrized basis.     |
+------------------------------+-----------------------------------+
| :py:meth:`spin_sym_map`      | Map from spin-j to 2j symmetrized |
|                              | qubits.                           |
+------------------------------+-----------------------------------+
| :py:meth:`spin_sym`          | Spin-j to 2j symmetrized qubits,  |
| :py:meth:`sym_spin`          | and back.                         |
+------------------------------+-----------------------------------+
| :py:meth:`perm_parity`       | Parity of a permutation.          |
| :py:meth:`antisymmetrize`    | Antisymmetrizes list of states.   |
+------------------------------+-----------------------------------+

"""

import numpy as np
import qutip as qt
from itertools import permutations, product

from spheres import *

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

def symmetrized_basis(n, d=2):
    """
    Constructs a symmetrized basis set for n systems in d dimensions.

    Parameters
    ----------
        n : int
            The number of systems to symmetrize.

        d : int or list
            Either an integer representing the dimensionality
            of the individual subsystems, in which case, we work 
            in the computational basis; or else a list of basis
            states for the individual systems.

    Returns
    -------
        sym_basis : dict
            ``sym_basis["labels"]`` is a list of labels for
            the symmetrized basis states. Each element of the list
            is a tuple whose length is the dimensionality of the
            individual subsystems, with an integer counting the number
            of subsystems in that basis state.

            ``sym_basis["basis"]`` is a dictionary mapping 
            labels to symmetrized basis states.

            ``sym_basis["map"]`` is a linear transformation
            from the permutation symmetric subspace to the
            full tensor product of the n systems.

            The dimensionality of the symmetric subspace
            corresponds to the number of ways of distributing
            :math:`n` elements in :math:`d` boxes, where :math:`n` is the number of
            systems and :math:`d` is the dimensionality of
            an individual subsytems. In other words, the dimensionality
            :math:`s` of the permutation symmetric subspace is :math:`\\binom{d+n-1}{n}`.

            So ``sym_basis["map"]`` is a map from
            :math:`\\mathbb{C}^{s} \\rightarrow \\mathbb{C}^{d^{n}}`.

    """
    if type(d) == int:
        d = [qt.basis(d, i) for i in range(d)]
    labels = list(filter(lambda b: sum(b) == n,\
                list(product(range(n+1), repeat=len(d)))[::-1]))
    sym_basis = dict([(label, symmetrize(flatten([[d[i]]*b \
                                    for i, b in enumerate(label)])))\
                                        for label in labels])
    sym_map = qt.Qobj(np.vstack([components(sym_basis[label])\
                                        for label in labels]).T)
    sym_map.dims =[[d[0].shape[0]]*n, [len(sym_basis)]]
    return {"labels": labels,\
            "basis": sym_basis,\
            "map": sym_map}


def spin_sym_map(j):
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

def spin_sym(spin, map=None):
    """
    Converts a spin-j state into a state of 2j symmetrized qubits. Constructs
    the linear map if not provided.
    """
    j = (spin.shape[0]-1)/2
    if not map:
        map = spin_sym_map(j)
    return map*spin

def sym_spin(sym, map=None):
    """
    Converts a state of 2j symmetrized qubits into a spin-j state. Constructs
    the linear map if not provided.
    """
    j = len(sym.dims[0])/2
    if not map:
        map = spin_sym_map(j)
    return map.dag()*sym

def perm_parity(lst):
    '''\
    Given a permutation of the digits 0..N in order as a list, 
    returns its parity (or sign): +1 for even parity; -1 for odd.

    https://code.activestate.com/recipes/578227-generate-the-parity-or-sign-of-a-permutation/
    '''
    lst = list(lst)
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity   

def antisymmetrize(pieces):
    """
    Antisymmetrizes provided list of states.
    """
    perms = list(permutations(list(range(len(pieces)))))
    anti = sum([perm_parity(perm)*qt.tensor(*[pieces[i] for i in perm])\
                for perm in perms])
    return anti.unit() if anti.norm() != 0 else anti