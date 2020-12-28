"""
Miscellaneous useful functions.
"""

import numpy as np
import qutip as qt
from itertools import *
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

def phase_angle(q):
    return np.mod(np.angle(phase(q)), 2*np.pi)

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
    return qt.tensor(*[qt.basis(dims[i], int(b))\
                for i, b in enumerate(bitstring)])

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

def qubits_xyz(state):
    """
    Qutip tensor state of qubits to their XYZ expectation values.
    """
    xyzs = []
    for i in range(len(state.dims[0])):
        dm = state.ptrace(i)
        xyz = np.array([qt.expect(qt.sigmax(), dm),\
                        qt.expect(qt.sigmay(), dm),\
                        qt.expect(qt.sigmaz(), dm)])
        xyzs.append(xyz)
    return np.array(xyzs)

def spinj_xyz(state):
    j = (state.shape[0]-1)/2
    return np.array([qt.expect(qt.jmat(j, 'x'), state),\
                     qt.expect(qt.jmat(j, 'y'), state),\
                     qt.expect(qt.jmat(j, 'z'), state)])

def pauli_basis(n):
    """
    Generates the Pauli basis for n qubits. Returns a dictionary associating
    a Pauli string (e.g., "IXY") to the tensor product of the corresponding Pauli operators.
    """
    IXYZ = {"I": qt.identity(2),\
            "X": qt.sigmax(),\
            "Y": qt.sigmay(),\
            "Z": qt.sigmaz()}
    return dict([("".join(pauli_str),\
                qt.tensor(*[IXYZ[o] for o in pauli_str]))\
                    for pauli_str in product(IXYZ.keys(), repeat=n)])

def to_pauli_basis(qobj, basis=None):
    """
    Expands a state/operator in the Pauli basis. Returns a dictionary associating
    a Pauli string to the corresponding component.
    """
    if basis == None:
        basis = pauli_basis(len(qobj.dims[0]))
    return dict([(pauli_str, qt.expect(pauli_op, qobj))\
                    for pauli_str, pauli_op in basis.items()])

def from_pauli_basis(exps, basis=None):
    """
    Given a dictionary mapping Pauli strings to components, returns the corresponding
    density matrix/operator.
    """
    if basis == None:
        basis = pauli_basis(len(list(exps.keys())[0]))
    n = len(list(basis.values())[0].dims[0])
    return sum([exps[pauli_str]*pauli_op/(2**n)
                    for pauli_str, pauli_op in basis.items()])

def random_pairs(n):
    """
    Generates a random list of pairs of n elements. 
    """
    pairs = list(combinations(list(range(n)), 2))
    np.random.shuffle(pairs) 
    final_pairs = []
    for p in pairs:
        pair = np.array(p)
        np.random.shuffle(pair)
        final_pairs.append(pair)
    return final_pairs 

def random_unique_pairs(n):
    """
    Generates a random list of pairs of n elements with no pair sharing an element.
    """
    pairs = list(combinations(list(range(n)), 2))
    pick = pairs[np.random.choice(len(pairs))]
    good_pairs = [pick]
    used = [*pick]
    def clean_pairs(pairs, used):
        to_remove = []
        for i, pair in enumerate(pairs):
            for u in used:
                if u in pair:
                    to_remove.append(i)
        return [i for j, i in enumerate(pairs) if j not in to_remove]
    pairs = clean_pairs(pairs, used)
    while len(pairs) > 0 and len(pairs) != clean_pairs(pairs, used):
        pick = pairs[np.random.choice(len(pairs))]
        good_pairs.append(pick)
        used.extend(pick)
        pairs = clean_pairs(pairs, used)
    return good_pairs

def tensor_upgrade(O, i, n):
    """
    Upgrades an operator to act on the i'th subspace of n.
    """
    return qt.tensor(*[O if i==j else qt.identity(O.shape[0]) for j in range(n)])

def dirac(state, probabilities=True):
    """
    Prints a pretty representation of a state in Dirac braket notation.
    """
    v = components(state)
    for i, bits in enumerate(product(*[list(range(d)) for d in state.dims[0]])):
        basis_str = "|%s>" % "".join([str(b) for b in bits])
        if not np.isclose(v[i], 0):
            if probabilities:
                print("%s: %.3f" % (basis_str, abs(v[i])**2))
            else:
                if np.isclose(v[i].imag, 0):
                    print("%.3f %s Pr: %.3f" % (v[i].real, basis_str, abs(v[i])**2))
                elif np.isclose(v[i].real, 0):
                    print("%.3fi %s Pr: %.3f" % (v[i].imag, basis_str, abs(v[i])**2))
                else:
                    print("%.3f+%.3fi %s Pr: %.3f" % (v[i].real, v[i].imag, basis_str, abs(v[i])**2))

