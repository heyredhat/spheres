"""
Miscellaneous useful functions.
"""

import numpy as np
import qutip as qt
from itertools import *
factorial = np.math.factorial

def rand_c(n=1):
    """
    Generates (n) random extended complex coordinate(s) whose real and imaginary parts
    are normally distributed, and ten percent of the time, we return :math:`\\infty`.

    Parameters
    ----------
        n : int
            Number of coordinates.
    
    Returns
    -------
        c : complex/inf or np.array
            n extended complex coordinates.
    """
    if n == 1:
        return np.random.randn() + 1j*np.random.randn() if np.random.random() > 0.1 else np.inf
    return np.array([np.random.randn() + 1j*np.random.randn() if np.random.random() > 0.1 else np.inf for i in range(n)])

def rand_xyz(n=1):
    """
    Generates (n) random point(s) on the unit sphere in cartesian coordinates.

    Parameters
    ----------
        n : int
            Number of coordinates.
    
    Returns
    -------
        xyz : np.array
            n cartesian coordinates.
    """
    if n == 1:
        return normalize(np.random.randn(3))
    return np.array([normalize(np.random.randn(3)) for i in range(n)])

def rand_sph(n=1):
    """
    Generates (n) random point(s) on the unit sphere in spherical coordinates.

    Parameters
    ----------
        n : int
            Number of coordinates.
    
    Returns
    -------
        xyz : np.array
            n spherical coordinates.
    """
    if n == 1:
        return np.array([np.pi*np.random.random(),\
                         2*np.pi*np.random.random()])
    else:
        return np.array([[np.pi*np.random.random(),\
                         2*np.pi*np.random.random()] for i in range(n)])

def components(q):
    """
    Extracts components of qt.Qobj, whether bra or ket, as a numpy array.

    Parameters
    ----------
        q : qt.Qobj 
            Qutip state.

    Returns
    -------
        n : np.array
            Numpy array.
    """
    return (q.full().T[0] if q.type == "ket" else q.full()[0])\
                if type(q) == qt.Qobj else q

def normalize(v):
    """
    Normalizes numpy vector. Simply passes the vector through if norm is 0.

    Parameters
    ----------
        v : np.array
            Numpy vector.

    Returns
    -------
        v : np.array
            Normalized numpy vector.
    """
    n = np.linalg.norm(v)
    return v/n if not np.isclose(n, 0) else v

def phase(v):
    """
    Extracts phase of a complex vector (np.ndarray or qt.Qobj) by finding
    the first non-zero component and returning its phase.

    Parameters
    ----------
        v : np.array or qt.Qobj
    
    Returns
    -------
        phase : complex
    """
    v = components(v) if type(v) == qt.Qobj else v
    return np.exp(1j*np.angle(v[(v!=0).argmax(axis=0)]))

def phase_angle(q):
    """
    Extracts phase angle of a complex vector (np.ndarray or qt.Qobj) by finding
    the first non-zero component and returning its phase angle.

    Parameters
    ----------
        v : np.array or qt.Qobj
    
    Returns
    -------
        phase_angle : float
    """
    return np.mod(np.angle(phase(q)), 2*np.pi)

def normalize_phase(v):
    """
    Normalizes the phase of a complex vector (np.ndarray or qt.Qobj).

    Parameters
    ----------
        v : np.array or qt.Qobj
    
    Returns
    -------
        v : np.array or qt.Qobj
            Phase normalized state.
    """
    return v/phase(v)

def compare_unordered(A, B, decimals=5):
    """
    Compares two sets of vectors regardless of their ordering up to some precision.

    Parameters
    ----------
        A : list
        
        B : list

        decimals : int

    Returns
    -------
        equal : bool
    """
    A, B = np.around(A, decimals=decimals), np.around(B, decimals=decimals)
    return np.array([a in B for a in A]).all()

def compare_nophase(a, b):
    """
    Compares two vectors disregarding their overall complex phase. 

    Parameters
    ----------
        a : qt.Qobj or np.array
        
        b : qt.Qobj or np.array

    Returns
    -------
        equal : bool
    """
    return np.allclose(normalize_phase(a), normalize_phase(b))

def compare_spinors(A, B, decimals=5):
    """
    Compares two lists of spinors, disregarding both their phases,
    as well as their ordering in the list.

    Parameters
    ----------
        A : list
        
        B : list

        decimals : int

    Returns
    -------
        equal : bool
    """
    A = np.array([components(normalize_phase(a)) for a in A])
    B = np.array([components(normalize_phase(b)) for b in B])
    return compare_unordered(A, B, decimals=decimals)

"""
:math:`SO(3)` generators :math:`L_{x}, L_{y}, L_{z}`.
"""
so3_generators = {"x": np.array([[0,0,0],\
                                 [0,0,-1],\
                                 [0,1,0]]),\
                  "y": np.array([[0,0,1],\
                                 [0,0,0],\
                                 [-1,0,0]]),\
                  "z": np.array([[0,-1,0],\
                                 [1,0,0],\
                                 [0,0,0]])}


def bitstring_basis(bitstring, dims=2):
    """
    Generates a basis vector corresponding to a given bitstring, 
    which may be a list of integers or a string of integers. The
    dimensionality of each tensor factor is given by dims, which may
    be either an integer (all the same dimensionality) or a list 
    (a dimension for each factor).

    Parameters
    ----------
        bitstring : str

        dims : int or list

    Returns
    -------
        tensor_state : qt.Qobj
    """
    if type(bitstring) == str:
        bitstring = [int(s) for s in bitstring]
    if type(dims) != list:
        dims = [dims]*len(bitstring)
    return qt.tensor(*[qt.basis(dims[i], int(b))\
                for i, b in enumerate(bitstring)])

def fix_stars(old_stars, new_stars):
    """
    Try to adjust the ordering of a list of stars to keep continuity, so that they
    are in the "same order." Not always reliable.

    Parameters
    ----------
        old_stars : list
            List of xyz coordinates.

        new_stars : list
            List of xyz coordinates.

    Returns
    -------
        fixed_stars : list
            List of xyz coordinates.
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
    Flattens list of lists.

    Parameters
    ----------
        to_flatten : list
            List of lists.

    Returns
    -------
        flattened : list
            Flattened list.
    """
    return [item for sublist in to_flatten for item in sublist]

def binomial(n, k):
    """
    Binomial coefficient :math:`\\binom{n}{k} = \\frac{n!}{k!(n-k)!}`

    Parameters
    ----------
        n : int
        k : int

    Returns
    -------
        binomial_coefficient : int
    """
    return int(factorial(n)/(factorial(k)*factorial(n-k)))

def qubits_xyz(state):
    """
    Returns XYZ expectation values for each qubit in a tensor product.

    Parameters
    ----------
        state : qt.Qobj
            Tensor state of qubits.

    Returns
    -------
        xyz : np.array
            Array of xyz coordinates.
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
    """
    Returns XYZ expectation values for a spin-j state.

    Parameters
    ----------
        state : qt.Qobj
            Spin-j state.

    Returns
    -------
        xyz : np.array
            XYZ expectation values.
    """
    j = (state.shape[0]-1)/2
    if j == 0:
        return np.array([0,0,0])
    return np.array([qt.expect(qt.jmat(j, 'x'), state),\
                     qt.expect(qt.jmat(j, 'y'), state),\
                     qt.expect(qt.jmat(j, 'z'), state)])

def pauli_basis(n):
    """
    Generates the Pauli basis for n qubits. Returns a dictionary associating
    a Pauli string (e.g., "IXY") to the tensor product of the corresponding Pauli operators.

    Parameters
    ----------
        n : int
            n qubits.

    Returns
    -------
        basis : dict
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

    Parameters
    ----------
        qobj : qt.Qobj
            State/operator
    
    Returns
    -------
        coeffs : dict
    """
    if basis == None:
        basis = pauli_basis(len(qobj.dims[0]))
    return dict([(pauli_str, qt.expect(pauli_op, qobj))\
                    for pauli_str, pauli_op in basis.items()])

def from_pauli_basis(coeffs, basis=None):
    """
    Given a dictionary mapping Pauli strings to components, returns the corresponding
    density matrix/operator.

    Parameters
    ----------
        exps : dict

    Returns
    -------
        operator : qt.Qobj
    """
    if basis == None:
        basis = pauli_basis(len(list(coeffs.keys())[0]))
    n = len(list(basis.values())[0].dims[0])
    return sum([coeffs[pauli_str]*pauli_op/(2**n)
                    for pauli_str, pauli_op in basis.items()])

def tensor_upgrade(O, i, n):
    """
    Upgrades an operator to act on the i'th subspace of n subsystems.

    Parameters
    ----------
        O : qt.Qobj
            Operator.
        i : int
            Which subsytem to act on.
        n : int
            Of how many.

    Returns
    -------
        upgraded : qt.Qobj
    """
    return qt.tensor(*[O if i==j else qt.identity(O.shape[0]) for j in range(n)])

def dirac(state, probabilities=False):
    """
    Prints a pretty representation of a state in Dirac braket notation.

    Parameters
    ----------
        state : qt.Qobj
        
        probabilities : bool
            If True, returns only the probabilities.
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

def density_to_purevec(density):
    """
    Converts a density matrix to a pure vector if it's rank-1.

    Parameters
    ----------
        density : qt.Qobj
    
    Returns
    -------
        pure_state : qt.Qobj

    """
    entropy = qt.entropy_vn(density) 
    if np.isclose(entropy, 0):
        U, S, V = np.linalg.svd(density.full())
        s = S.tolist()
        for i in range(len(s)):
            if np.isclose(s[i], 1):
                return qt.Qobj(np.conjugate(V[i]))

def polygon_area(phis, thetas, radius = 1):
    """
    Computes area of spherical polygon.
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.

    Thanks to https://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python

    Parameters
    ----------
        phis : list

        thetas : list

        radius : float

    Returns
    -------
        area : float
    """
    from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
    lats = thetas - np.pi/2
    lons = phis
    #lats = np.deg2rad(lats)
    #lons = np.deg2rad(lons)

    # Line integral based on Green's Theorem, assumes spherical Earth

    #close polygon
    #if lats[0]!=lats[-1]:
    #    lats = append(lats, lats[0])
    #    lons = append(lons, lons[0])

    #colatitudes relative to (0,0)
    a = sin(lats/2)**2 + cos(lats)* sin(lons/2)**2
    colat = 2*arctan2( sqrt(a), sqrt(1-a) )

    #azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas=diff(colat)/2
    colat=colat[0:-1]+deltas

    # Perform integral
    integrands = (1-cos(colat)) * daz

    # Integrate 
    area = abs(sum(integrands))/(4*pi)

    area = min(area,1-area)
    if radius is not None: #return in units of radius
        return area * 4*pi*radius**2
    else: #return in ratio of sphere total area
        return area
        
def measure(state, projectors):
    """
    Given a state and a set of projectors, calculates the probability of each outcome, and returns an outcome index 
    with that probability.

    Parameters
    ----------
        state : qt.Qobj

        projectors : list

    Returns
    -------
        index : int
    """
    probs = np.array([qt.expect(proj, state) for proj in projectors])
    return np.random.choice(list(range(len(probs))), p=abs(probs/sum(probs)))
