"""
Symmetrization
--------------

"""

import numpy as np
import qutip as qt
from itertools import permutations, product

from pytket import Circuit
from pytket.circuit import Unitary1qBox, Unitary2qBox

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

def prepare_qubits(xyzs):
    circ = Circuit(len(xyzs))
    for i, xyz in enumerate(xyzs):
        r, phi, theta = xyz_sph(xyz)
        circ.Ry(theta/np.pi, i)
        circ.Rz(phi/np.pi, i)
    return circ

def Rk(k, dagger=False):
    M = (1/np.sqrt(k+1))*np.array([[1, -np.sqrt(k)],\
                                     [np.sqrt(k), 1]])
    return Unitary1qBox(M if not dagger else M.T)

def Tkj(k, j, dagger=False):
    M = (1/np.sqrt(k-j+1))*np.array([[np.sqrt(k-j+1), 0, 0, 0],\
                                       [0, 1, np.sqrt(k-j), 0],\
                                       [0, -np.sqrt(k-j), 1, 0],\
                                       [0, 0, 0, np.sqrt(k-j+1)]])
    return Unitary2qBox(M if not dagger else M.T)

def prepare_spin(spin, measure_cntrls=True):
    j = (spin.shape[0]-1)/2
    n = int(2*j)
    r = int(n*(n-1)/2)

    circ = prepare_qubits(spin_xyz(spin))
    spin_qubits = circ.qubits
    cntrl_qubits = circ.add_q_register("cntrls", r)
    cntrl_bits = circ.add_c_register("c_cntrls", r)
    qubits = circ.qubits

    offset = r
    for k in range(1, n):
        offset = offset-k
        circ.add_unitary1qbox(Rk(k),\
                              qubits[offset])
        for i in range(k-1):
            circ.add_unitary2qbox(Tkj(k, i+1),\
                                  qubits[offset+i+1],\
                                  qubits[offset+i])
        for i in range(k-1, -1, -1):
            circ.CSWAP(qubits[offset+i],\
                       qubits[r+k],\
                       qubits[r+i])
        for i in range(k-2, -1, -1):
            circ.add_unitary2qbox(Tkj(k, i+1, dagger=True),\
                                  qubits[offset+i+1],\
                                  qubits[offset+i])    
        circ.add_unitary1qbox(Rk(k, dagger=True),\
                              qubits[offset])
    if measure_cntrls:
        for i in range(r):
            circ.Measure(cntrl_qubits[i], cntrl_bits[i])
    return {"spin_qubits": spin_qubits,\
            "cntrl_qubits": cntrl_qubits,\
            "cntrl_bits": cntrl_bits,\
            "circuit": circ}

def tomography_circuits(circuit, on_qubits=None):
    on_qubits = circuit.qubits if on_qubits == None else on_qubits
    n_qubits = len(on_qubits)
    IXYZ = ["I", "X", "Y", "Z"]
    circuits = []
    for pauli_str in product(IXYZ, repeat=n_qubits):
        circ = circuit.copy()
        tomog_bits = circ.add_c_register("tomog", n_qubits)
        for i, o in enumerate(pauli_str):
            if o == "Y":
                circ.Rx(1/2, on_qubits[i])
                circ.Measure(on_qubits[i], tomog_bits[i])
            elif o == "X":
                circ.Ry(-1/2, on_qubits[i])
                circ.Measure(on_qubits[i], tomog_bits[i])
            elif o == "Z":
                circ.Measure(on_qubits[i], tomog_bits[i])
        circuits.append({"pauli": "".join(pauli_str),\
                         "circuit": circ,\
                         "tomog_bits": tomog_bits})
    return circuits

def dists_pauli_expectations(tomog_circs, dists):
    return dict([(tomog_circs[i]["pauli"], \
                 sum([prob*np.prod([1 if b == 0 else -1 \
                                        for b in bitstr])\
                        for bitstr, prob in dist.items()]))
                    for i, dist in enumerate(dists)])

def postselected_dists(results, on):
    dists = []
    for result in results:
        cbit_indices = result.c_bits
        dist = result.get_distribution()
        new_dist_bit_strs = []
        new_dist_probs = []
        for bit_str, prob in dist.items():
            good = True
            remove_indices = []
            for cntrl_cbit, post_val in on.items():
                if bit_str[cbit_indices[cntrl_cbit]] != post_val:
                    good = False
                    break
                else:
                    remove_indices.append(cbit_indices[cntrl_cbit])
            if good:
                bit_str = np.array(bit_str)
                new_dist_bit_strs.append(tuple(np.delete(bit_str, remove_indices)))
                new_dist_probs.append(prob)
        new_dist_probs = np.array(new_dist_probs)/sum(new_dist_probs)
        new_dist = dict(zip(new_dist_bit_strs, new_dist_probs))
        dists.append(new_dist)
    return dists
