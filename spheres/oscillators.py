"""
Oscillators
-----------

+---------------------------------------+----------------------------------------------------------------+
| :py:meth:`vacuum`                     | Constructs vacuum state.                                       |
| :py:meth:`annihilators`               | Constructs oscillator                                          |
|                                       | annihilators.                                                  |
+---------------------------------------+----------------------------------------------------------------+
| :py:meth:`second_quantize_operator`   | Second quantizes operator.                                     |
| :py:meth:`second_quantize_state`      | Creation operator for second quantized state.                  |
| :py:meth:`second_quantize_spin_state` | Creation operator for second quantized spin state.             |
+---------------------------------------+----------------------------------------------------------------+
| :py:meth:`osc_spin_permutation        | Permutation operator from tensor basis to direct sum of spins. |
| :py:meth:`extract_osc_spinblocks      | Extracts spin-j blocks from an operator.                       |
| :py:meth:`extract_osc_spinstates      | Extracts spin-j states from a state.                           |
+---------------------------------------+----------------------------------------------------------------+
| :py:meth:`spin_osc                    | Converts from a spin-j state to a 2D oscillator state.         |
+---------------------------------------+----------------------------------------------------------------+

"""

from spheres import *

def vacuum(n=2, max_ex=3):
    """
    Constructs vacuum state for a given number of oscillators with given cutoff dimension.
    """
    vac = qt.basis(max_ex**n)
    vac.dims = [[max_ex]*n, [1]*n]
    return vac

def annihilators(n=2, max_ex=3):
    """
    Constructs annihilators for a given number of oscillators with given cutoff dimension.
    """
    return [qt.tensor(*[qt.destroy(max_ex) if i == j\
                            else qt.identity(max_ex)\
                                    for j in range(n)])\
                                        for i in range(n)]

def second_quantize_operator(O, a):
    """
    Upgrades a first quantized operator to a second quantized operator given a list of annihilators.
    """
    O = O.full()
    terms = []
    for i in range(len(a)):
        for j in range(len(a)):
            terms.append(a[i].dag()*O[i][j]*a[j])
    return sum(terms)

def second_quantize_state(q, a):
    """
    Upgrades a first quantized state to a second quantized creation operator given a list of annihilators.
    """
    q = components(q)
    return sum([q[i]*a[i].dag() for i in range(len(a))])

def second_quantize_spin_state(spin, a):
    """
    Upgrades a spin state to a second quantized operator given a list of annihilators.
    """
    n = spin.shape[0]-1
    j = (spin.shape[0]-1)/2.
    v = spin.full().T[0]
    terms = []
    z, w = [a_.dag() for a_ in a]
    for m in np.arange(-j, j+1, 1):
        i = int(m+j)
        terms.append(v[i]*(z**(n-i))*(w**i)*\
                        (np.sqrt(factorial(2*j)/\
                            (factorial(j-m)*factorial(j+m)))))
    return sum(terms)

def osc_spin_permutation(max_ex):    
    """
    Returns permutation from the tensor basis of two oscillators to the basis organized by total N.
    """
    tensor_basis_labels = list(product(list(range(max_ex)), repeat=2))
    full_total_n_basis_labels = []
    for i in range(2*max_ex):
        full_total_n_basis_labels.extend([(i-j-1, j) for j in range(i)])
    n = len(full_total_n_basis_labels)
    m = max_ex**2
    P = np.zeros((n, m))
    for i, label in enumerate(tensor_basis_labels):
        P[full_total_n_basis_labels.index(label)][i] = 1
    P = qt.Qobj(P)
    P.dims = [[n], [max_ex, max_ex]]
    sums = [sum(label) for label in full_total_n_basis_labels]
    unique_sums = set(sums)
    dims = [sums.count(us) for us in unique_sums]
    return P, dims

def extract_osc_spinblocks(O, P=None, dims=None):
    """
    Extracts spin-j blocks from a 2D oscillator operator.
    """
    max_ex = int(np.sqrt(O.shape[0]))
    if not P:
        P, dims = osc_spin_permutation(max_ex)
    M = (P*O*P.dag()).full()
    running, blocks = 0, []
    for d in dims:
        blocks.append(qt.Qobj(M[running:running+d, running:running+d]))
        running += d
    return blocks

def extract_osc_spinstates(q, P=None, dims=None):
    """
    Extracts spin-j states from a 2D oscillator state.
    """
    max_ex = int(np.sqrt(q.shape[0]))
    if not P:
        P, dims = osc_spin_permutation(max_ex)
    v = (P*q).full().T[0]
    running, blocks = 0, []
    for d in dims:
        blocks.append(qt.Qobj(v[running:running+d]))
        running += d
    return blocks

def spin_osc(spin, max_ex=None):
    """
    Returns the 2D oscillator state corresponding to a given spin-j state.
    """
    j = (spin.shape[0]-1)/2
    if not max_ex:
        max_ex = int(2*j)
    vac = vacuum(n=2, max_ex=max_ex)
    a = annihilators(n=2, max_ex=max_ex)
    return (second_quantize_spin_state(spin, a)*vac).unit()

def second_quantized_paulis(max_ex=3):
    a = annihilators(n=2, max_ex=max_ex)
    return {"X": second_quantize_operator(qt.sigmax()/2, a),\
            "Y": second_quantize_operator(qt.sigmay()/2, a),\
            "Z": second_quantize_operator(qt.sigmaz()/2, a)}

def osc_spinj_xyz(osc):
    paulis = second_quantized_paulis(max_ex=osc.dims[0][0])
    return np.array([qt.expect(paulis[o], osc) for o in ["X", "Y", "Z"]])