"""
Functions for dealing with oscillators, particularly in the case
of double oscillators in the context of the Schwinger representation of spin.
"""

from scipy.special import binom
from scipy.linalg import block_diag
from .utils import *

def vacuum(n=2, cutoff_dim=3):
    """
    Constructs vacuum state for a given number of oscillators with given cutoff dimension.
    
    Parameters
    ----------
        n : int
            Number of oscillators.
        
        cutoff_dim : int
            Cutoff for the Fock space of the oscillators.
    
    Returns
    -------
        vac : qt.Qobj
            Vacuum state with the right tensor dimensions.
    """
    vac = qt.basis(cutoff_dim**n)
    vac.dims = [[cutoff_dim]*n, [1]*n]
    return vac

def annihilators(n=2, cutoff_dim=3):
    """
    Constructs annihilators for a given number of oscillators with given cutoff dimension.
    
    Parameters
    ----------
        n : int
            Number of oscillators.
        
        cutoff_dim : int
            Cutoff for the Fock space of the oscillators.
    
    Returns
    -------
        a : list
            List of annihilators.
    """
    return [tensor_upgrade(qt.destroy(cutoff_dim), i, n) for i in range(n)]

def second_quantize_operator(O, a=None):
    """
    Upgrades a first quantized operator to a second quantized operator given a list of annihilators.
    If no annihilators provided, it constructs them.

    Parameters
    ----------
        O : qt.Qobj
            First quantized operator.
        
        a : list
            List of annihilators.
    
    Returns
    -------
        OO : qt.Qobj
            Second quantized operator.
    """
    a = a if a else annihilators(n=O.shape[0])
    O = O.full()
    return sum([a[i].dag()*O[i][j]*a[j] for i in range(len(a)) for j in range(len(a))])

def second_quantize_state(q, a=None, state=False):
    """
    Upgrades a first quantized state to a second quantized creation operator given a list of annihilators. 
    If the annihilators aren't provided, they are constructed.

    Parameters
    ----------
        q : qt.Qobj
            First quantized state.
        
        a : list
            List of annihilators.

        state : bool
            If True, returns the second quantized state itself, obtained by acting with 
            the creation operator on the vacuum.
    
    Returns
    -------
        Q : qt.Qobj
            Second quantized creation operator (or state).
    """
    a = a if a else annihilators(n=q.shape[0])
    q = components(q)
    O = sum([q[i]*a[i].dag() for i in range(len(a))])
    return O if not state else O*vacuum(n=len(a), cutoff_dim=a[0].dims[0][0])

def second_quantize_spin_state(spin, a=None):
    """
    Upgrades a spin state to a second quantized creation operator given a list of annihilators.
    If the annihilators aren't provided, they are constructed.

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.
        
        a : list
            List of annihilators.
    
    Returns
    -------
        S : qt.Qobj
            Second quantized creation operator for the constellation.
    """
    a = a if a else annihilators(cutoff_dim=spin.shape[0]+1)
    j = (spin.shape[0]-1)/2.
    v = components(spin)
    z, w = [a_.dag() for a_ in a]
    return sum([np.sqrt(binom(int(2*j),int(j-m)))*v[int(j+m)]*z**(j-m)*w**(j+m) for m in np.arange(-j, j+1, 1)])

def osc_spintower_map(cutoff_dim):    
    """
    Returns permutation from the tensor basis of two oscillators to the basis organized by total N, in 
    other words, to a tower of spin states. Automatically padded so that higher spins whose full Hilbert space
    is truncated by the cutoff dimension have the right dimensionality.

    Parameters
    ----------
        cutoff_dim : int
            Cutoff dimension of the Fock spaces.
    
    Returns
    -------
        P : qt.Qobj
            Permutation operator.
    """
    tensor_basis_labels = list(product(list(range(cutoff_dim)), repeat=2))
    full_total_n_basis_labels = []
    for i in range(2*cutoff_dim):
        full_total_n_basis_labels.extend([(i-j-1, j) for j in range(i)])
    n, m = len(full_total_n_basis_labels), cutoff_dim**2
    P = np.zeros((n, m))
    for i, label in enumerate(tensor_basis_labels):
        P[full_total_n_basis_labels.index(label)][i] = 1
    P = qt.Qobj(P)
    P.dims = [[n], [cutoff_dim, cutoff_dim]]
    return P

def spin_tower_dimensions(d):
    """
    Given the overal dimension of a spin tower, return 
    the individual dimensions of the spin states.
    E.g., 15 = 1 + 2 + 3 + 4 + 5

    Parameters
    ----------
        d : int
            Overall dimension.

    Returns
    -------
        dims : list
            Individual dimensions.
    """
    dims = [1]
    while sum(dims) != d:
        dims.append(dims[-1]+1)
    return dims

def osc_spinblocks(O, map=None):
    """
    Extracts spin-j blocks from a 2D oscillator operator. 

    Parameters
    ----------
        O : qt.Qobj
            2D oscillator operator.
        
        map : qt.Qobj
            Map from tensor basis to the spin tower basis.
            Automatically constructed if not provided.
    
    Returns
    -------
        blocks : list
            List of qt.Qobj operators appearing along the diagonal.
    """
    cutoff_dim = int(np.sqrt(O.shape[0]))
    map = map if map else osc_spintower_map(cutoff_dim)
    dims = spin_tower_dimensions(map.shape[0])
    M = (map*O*map.dag()).full()
    running, blocks = 0, []
    for d in dims:
        blocks.append(qt.Qobj(M[running:running+d, running:running+d]))
        running += d
    return blocks

def spin_osc_map(j, cutoff_dim=None):
    """
    Construct linear map from spin-j states into the Fock space of the 2D quantum harmonic oscillator.

    Parameters
    ----------
        j : float
            j-value of the spin.
        
        cutoff_dim : int
            Cutoff dimensions of the Fock space.

    Returns
    -------
        map : qt.Qobj
            Linear map from spin-j Hilbert space to the Fock space of the 2D oscillator.
    """
    cutoff_dim = cutoff_dim if cutoff_dim else int(2*j+1)
    return sum([qt.tensor(qt.basis(cutoff_dim, int(2*j-i)), qt.basis(cutoff_dim, i))*qt.spin_state(j, j-i).dag() for i in range(int(2*j+1))])

def osc_spins(q, map=None):
    """
    Extracts spin-j states from a 2D oscillator state.

    Parameters
    ----------
        q : qt.Qobj
            2D oscillator state.
        
        map : qt.Qobj
            Map from tensor basis to the spin tower basis.
            Automatically constructed if not provided.
    
    Returns
    -------
        blocks : list
            List of spins as qt.Qobj's.
    """
    j = (q.shape[0]-1)/2
    cutoff_dim = int(np.sqrt(q.shape[0]))
    map = map if map else osc_spintower_map(cutoff_dim)
    dims = spin_tower_dimensions(map.shape[0])
    v = (map*q).full().T[0]
    running, blocks = 0, []
    for d in dims:
        blocks.append(qt.Qobj(v[running:running+d]))
        running += d
    return blocks

def spin_osc(spin, cutoff_dim=None, map=None):
    """
    Returns the 2D oscillator state corresponding to a given spin-j state (pure or mixed).

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.
        
        cutoff_dim : int
            Cutoff dimension.

        map : qt.Qobj
            Map from spin-j Hilbert space to double harmonic oscillator space. Constructed if not provided.
    
    Returns
    -------
        osc : qt.Qobj
            2D quantum harmonic oscillator state.
    """
    j = (spin.shape[0]-1)/2
    cutoff_dim = cutoff_dim if cutoff_dim else int(2*j+1)
    if spin.type == 'oper':
        map = map if map else spin_osc_map(j, cutoff_dim=cutoff_dim)
        return map*spin*map.dag()
    vac = vacuum(n=2, cutoff_dim=cutoff_dim)
    a = annihilators(n=2, cutoff_dim=cutoff_dim)
    return (second_quantize_spin_state(spin, a)*vac).unit()

def spins_osc(spins, cutoff_dim=None, map=None):
    """
    List of spin-j states to a 2D quantum harmonic oscillator state.

    Parameters
    ----------
        osc : qt.Qobj
            Double oscillator state.

        cutoff_dim : int
            Cutoff dimension for the 2D oscillator Fock space.

        map : qt.Qobj
            Map from tensor basis to the spin tower basis.
            Automatically constructed if not provided.    
            
    Returns
    -------
        osc : qt.Qobj
            Double oscillator state.
    """
    cutoff_dim = cutoff_dim if cutoff_dim else int(np.ceil(spins[-1].shape[0]/2))
    map = map if map else osc_spintower_map(cutoff_dim=cutoff_dim)
    if spins[-1].type == "oper":
        dm = qt.Qobj(block_diag(*[spin.full() for spin in spins]))
        return map.dag()*dm*map
    else:
        state = qt.Qobj(np.concatenate([components(spin) for spin in spins]))
        return map.dag()*state

def osc_spin(osc, map=None):
    """
    Returns (nonzero) spin-j states correspond to the 2D oscillator state (pure or mixed).

    Parameters
    ----------
        osc : qt.Qobj
            Double oscillator state.

        map : qt.Qobj
            Map from tensor basis to the spin tower basis.
            Automatically constructed if not provided.    
    Returns
    -------
        spins : list
            List of spins.
    """
    spins = [spin for spin in osc_spins(osc, map=map) if spin.norm() != 0]
    return spins[0] if len(spins) == 1 else spins

def second_quantized_paulis(cutoff_dim=3):
    """
    Second quantized Pauli X, Y, Z operators on two harmonic oscillators.

    Parameters
    ----------
        cutoff_dim : int
            Cutoff dimensions for the oscillator Fock spaces.
    
    Returns
    -------
        XYZ : dict
            Dictionary of operators {"X": X, "Y": Y, "Z": Z}.

    """
    a = annihilators(n=2, cutoff_dim=cutoff_dim)
    return {"x": second_quantize_operator(qt.sigmax()/2, a),\
            "y": second_quantize_operator(qt.sigmay()/2, a),\
            "z": second_quantize_operator(qt.sigmaz()/2, a)}

def spinj_xyz_osc(osc, paulis=None):
    """
    <X>, <Y>, <Z> expectation values on the given double oscillator state.

    Parameters
    ----------
        osc : qt.Qobj
            Double oscillator state.

        paulis : dict
            Dictionary of second quantized Pauli's. Constructed if not provided.
    
    Returns
    -------
        xyz : np.ndarray
            Array of Pauli expectation values.
    """
    paulis = paulis if paulis else second_quantized_paulis(cutoff_dim=osc.dims[0][0])
    return np.array([qt.expect(paulis[o], osc) for o in ["x", "y", "z"]])

