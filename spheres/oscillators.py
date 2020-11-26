"""
Oscillators
--------
"""

from spheres import *

def vacuum(n=2, max_ex=3):
    vac = qt.basis(max_ex**n)
    vac.dims = [[max_ex]*n, [1]*n]
    return vac

def annihilators(n=2, max_ex=3):
    return [qt.tensor(*[qt.destroy(max_ex) if i == j\
                            else qt.identity(max_ex)\
                                    for j in range(n)])\
                                        for i in range(n)]

def second_quantize_operator(O, a):
    O = O.full()
    terms = []
    for i in range(len(a)):
        for j in range(len(a)):
            terms.append(a[i].dag()*O[i][j]*a[j])
    return sum(terms)

def second_quantize_state(q, a):
    q = components(q)
    return sum([q[i]*a[i].dag() for i in range(len(a))])

def second_quantize_spin_state(spin, a):
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
    tensor_basis_labels = list(product(list(range(max_ex)), repeat=2))
    full_total_n_basis_labels = []
    for i in range(2*max_ex):
        full_total_n_basis_labels.extend([(i-j-1, j) for j in range(i)])
    #total_n_basis_labels = [label for label in full_total_n_basis_labels if label in tensor_basis_labels]
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
    max_ex = int(np.sqrt(q.shape[0]))
    if not P:
        P, dims = osc_spin_permutation(max_ex)
    v = (P*q).full().T[0]
    running, blocks = 0, []
    for d in dims:
        blocks.append(qt.Qobj(v[running:running+d]))
        running += d
    return blocks