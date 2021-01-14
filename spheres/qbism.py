import qutip as qt
import numpy as np
import pkg_resources

def clock(d):
    return sum([qt.basis(d, i+1)*qt.basis(d, i).dag()\
                    if i != d-1 else qt.basis(d, 0)*qt.basis(d, i).dag()\
                        for i in range(d) for j in range(d)])/d

def shift(d):
    w = np.exp(2*np.pi*1j/d)
    return qt.Qobj(np.diag([w**i for i in range(d)]))

def displace(d, a, b):
    Z, X = clock(d), shift(d)
    return (-np.exp(1j*np.pi/d))**(a*b)*X**b*Z**a

def displacement_operators(d):
    return dict([((a, b), displace(d, a, b)) for b in range(d) for a in range(d)])

def load_fiducial(d):
    """
    http://www.physics.umb.edu/Research/QBism/solutions.html
    """
    f = pkg_resources.resource_stream(__name__, "sic_povms/d%d.txt" % d)
    #f = open("sic_povms/d%d.txt" % d, "r")
    fiducial = []
    for line in f:
        if line.strip() != "":
            re, im = [float(v) for v in line.split()]
            fiducial.append(re + 1j*im)
    return qt.Qobj(np.array(fiducial)).unit()

def load_sic_states(d):
    fiducial = load_fiducial(d)
    return [D*fiducial for index, D in displacement_operators(d).items()]

def sic_test(sic):
    d = int(np.sqrt(len(sic)))
    for i, s in enumerate(sic):
        for j, t in enumerate(sic):
            should_be = 1 if i == j else 1/(d+1)
            inner = np.abs(s.overlap(t)**2) if s.type == 'ket' else (s*t).tr()
            print("(%d, %d): %.4f | should be: %.4f" % (i, j, inner, should_be))

def sic_povm(sic_states):
    d = int(np.sqrt(len(sic_states)))
    projectors = [state*state.dag() for state in sic_states]
    return {"states": sic_states,\
            "projectors": projectors,\
            "elements": [(1/d)*projector for projector in projectors]}

def rho_prob(rho, povm):
    d = rho.shape[0]
    return np.array([(E*rho).tr() for E in povm["elements"]]).real

def prob_rho(p, povm):
    d = int(np.sqrt(len(p)))
    return sum([((d+1)*p[i] - 1/d)*povm["projectors"][i] for i in range(d**2)])

def prob_rho2(p, povm):
    d = int(np.sqrt(len(p)))
    return (d+1)*sum([p[i]*povm["projectors"][i] for i in range(d**2)]) - qt.identity(d)

def vn_conditional_probs(von_neumann, ref_povm):
    d = von_neumann.shape[0]
    vn_projectors = [v*v.dag() for v in von_neumann.eigenstates()[1]]
    return np.array([[(ref_povm["projectors"][i]*vn_projectors[j]).tr() for i in range(d**2)] for j in range(d)]).real

def vn_posterior(rho, von_neumann, ref_povm):
    d = rho.shape[0]
    p = rho_prob(rho, ref_povm)
    r = vn_conditional_probs(von_neumann, ref_povm)
    return np.array([sum([p[i]*r[j][i] for i in range(d**2)]) for j in range(d)])

def povm_conditional_probs(povm, ref_povm):
    d = int(np.sqrt(len(ref_povm["projectors"])))
    return np.array([[(ref_povm["projectors"][i]*povm["elements"][j]).tr() for i in range(d**2)] for j in range(d**2)]).real

def povm_posterior(rho, povm, ref_povm):
    d = rho.shape[0]
    p = rho_prob(rho, ref_povm)
    r = povm_conditional_probs(povm, ref_povm)
    return np.array([sum([p[i]*r[j][i] for i in range(d**2)]) for j in range(d**2)])

def vn_born(rho, von_neumann, ref_povm):
    d = rho.shape[0]
    p = rho_prob(rho, ref_povm)
    r = vn_conditional_probs(von_neumann, ref_povm)
    return np.array([(d+1)*sum([p[i]*r[j][i] for i in range(d**2)])-1 for j in range(d)]).real

def povm_born(rho, povm, ref_povm):
    d = rho.shape[0]
    p = rho_prob(rho, ref_povm)
    r = povm_conditional_probs(povm, ref_povm)
    return np.array([(d+1)*sum([p[i]*r[j][i] for i in range(d**2)]) - (1/d)*sum([r[j][i] for i in range(d**2)]) for j in range(d**2)]).real

def vn_born_matrix(rho, von_neumann, ref_povm):
    d = rho.shape[0]
    p = rho_prob(rho, ref_povm)
    r = vn_conditional_probs(von_neumann, ref_povm)
    M = (d+1)*np.eye(d**2) - (1/d)*np.ones((d**2,d**2))
    return r @ M @ p

def povm_born_matrix(rho, povm, ref_povm):
    d = rho.shape[0]
    p = rho_prob(rho, ref_povm)
    r = povm_conditional_probs(povm, ref_povm)
    M = (d+1)*np.eye(d**2) - (1/d)*np.ones((d**2,d**2))
    return r @ M @ p

def temporal_conditional_probs(U, ref_povm):
    d = U.shape[0]
    return np.array([[(1/d)*(U*ref_povm["projectors"][i]*U.dag()*ref_povm["projectors"][j]).tr() for i in range(d**2)] for j in range(d**2)]).real

def quantum_inner_product(p, s):
    d = int(np.sqrt(len(p)))
    return d*(d+1)*np.dot(p, s) - 1

def povm_implementation(povm):
    n = len(povm["elements"])
    d = int(np.sqrt(n))
    aux_projectors = [qt.tensor(qt.identity(d), qt.basis(n, i)*qt.basis(n, i).dag()) for i in range(n)]
    V = sum([qt.tensor(povm["elements"][i].sqrtm(), qt.basis(n, i)) for i in range(n)])
    povm_elements = [V.dag()*aux_projectors[i]*V for i in range(n)]
    assert np.all([np.allclose(povm["elements"][i], povm_elements[i]) for i in range(n)])
    Q, R = np.linalg.qr(V, mode="complete")
    for i in range(d):
        Q.T[[i,n*i]] = Q.T[[n*i,i]]
    U = qt.Qobj(-Q)
    U.dims = [[d, n],[d, n]]
    assert np.allclose(V, U*qt.tensor(qt.identity(d), qt.basis(n, 0)))
    return U