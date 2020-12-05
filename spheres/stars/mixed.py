import numpy as np
import qutip as qt

from spheres.utils import *

def spherical_tensor(j, sigma, mu):
    terms = []
    for m1 in np.arange(-j, j+1):
        for m2 in np.arange(-j, j+1):
            terms.append(\
                ((-1)**(j-m2))*\
                qt.clebsch(j, j, sigma, m1, -m2, mu)*\
                qt.spin_state(j, m1)*qt.spin_state(j, m2).dag())
    return sum(terms)

def spherical_tensor_basis(j):
    T_basis = {}
    for sigma in np.arange(0, int(2*j+1)):
        for mu in np.arange(-sigma, sigma+1):
            T_basis[(sigma, mu)] = spherical_tensor(j, sigma, mu)
    return T_basis

def operator_spherical_decomposition(O, T_basis=None):
    j = (O.shape[0]-1)/2
    if not T_basis:
        T_basis = spherical_tensor_basis(j)
    decomposition = {}
    for sigma in np.arange(0, int(2*j+1)):
        for mu in np.arange(-sigma, sigma+1):
            decomposition[(sigma, mu)] = (O*T_basis[(sigma, mu)].dag()).tr()
    return decomposition

def spherical_decomposition_operator(decomposition, T_basis=None):
    j = max([k[0] for k in decomposition.keys()])/2
    if not T_basis:
        T_basis = spherical_tensor_basis(j)
    terms = []
    for sigma in np.arange(0, int(2*j+1)):
        for mu in np.arange(-sigma, sigma+1):
            terms.append(decomposition[(sigma, mu)]*T_basis[(sigma, mu)])
    return sum(terms)

def spherical_decomposition_spins(decomposition):
    max_j = max([k[0] for k in decomposition.keys()])
    return [qt.Qobj(np.array([decomposition[(j, m)]\
                        for m in np.arange(j, -j-1, -1)]))\
                            for j in np.arange(0, max_j+1)]

def spins_spherical_decomposition(spins):
    max_j = (spins[-1].shape[0]-1)/2
    decomposition = {}
    for j in np.arange(0, max_j+1):
        for m in np.arange(j, -j-1, -1):
            decomposition[(j, m)] = components(spins[int(j)])[int(j-m)]
    return decomposition

def operator_spins(O, T_basis=None):
    return spherical_decomposition_spins(operator_spherical_decomposition(O, T_basis=T_basis))

def spins_operator(spins, T_basis=None):
    return spherical_decomposition_operator(spins_spherical_decomposition(spins), T_basis=T_basis)
