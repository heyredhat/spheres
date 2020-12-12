"""
Symplectic Transformations
--------------------------

"""
import numpy as np
import qutip as qt
import scipy as sc

def make_gaussian_operator(A, B=None, h=None):
    """
    Returns a random Gaussian operator in the form of a Hermitian matrix and a vector,
    to be second quantized.
    """
    B = np.zeros(A.shape) if type(B) == type(None) else B
    h = np.zeros(A.shape[0]) if type(h) == type(None) else h
    return np.block([[A, B],\
                     [B.conj(), A.conj()]]),\
           np.concatenate([h, h.conj()])

def random_gaussian_operator(n):
    """
    Returns random Gaussian operator in the form of a Hermitian matrix and a vector,
    to be second quantized.
    """
    A = qt.rand_herm(n).full()
    B = np.random.randn(n, n) + 1j*np.random.randn(n, n)
    B = B @ B.T + B.T @ B
    h = np.random.randn(n) + 1j*np.random.randn(n)
    return make_gaussian_operator(A, B, h)

def gaussian_complex_symplectic(H, h, expm=True, theta=2):
    """
    Converts a Gaussian operator (in the form of a Hermitian matrix and a vector)
    into a complex symplectic matrix/vector.
    """
    n = int(len(h)/2)
    omega = omega_c(n)
    S = sc.linalg.expm(-1j*(theta/2)*omega@H) if expm else H
    try:
        s = ((S - np.eye(2*n)) @ np.linalg.inv(H)) @ h
    except:
        s = ((S - np.eye(2*n)) @ np.linalg.pinv(H)) @ h
    return S, s

def omega_c(n):
    """
    Complex symplectic form.
    """
    return sc.linalg.block_diag(np.eye(n), -np.eye(n))

def test_c(S):
    """
    Test if an matrix is complex symplectic.
    """
    n = int(len(S)/2)
    WC = omega_c(n)
    return np.allclose(S @ WC @ S.conj().T, WC)
    
def complex_real_symplectic(S, s):
    """
    Converts a complex symplectic matrix/vector to a real symplectic matrix/vector.
    """
    n = int(len(s)/2)
    L = (1/np.sqrt(2))*np.block([[np.eye(n), np.eye(n)],\
                             [-1j*np.eye(n), 1j*np.eye(n)]])
    return (L @ S @ L.conj().T).real, (L @ s).real

def complex_real_symplectic2(S, s):
    """
    Converts a complex symplectic matrix/vector to a real symplectic matrix/vector.
    Alternative construction.
    """
    n = int(len(s)/2)
    E, F = S[0:n, 0:n], S[0:n, n:]
    return np.block([[(E+F).real, -(E-F).imag],\
                     [(E+F).imag, (E-F).real]]),\
           np.sqrt(2)*np.concatenate([s[0:n].real,\
                                      s[0:n].imag])

def omega_r(n):
    """
    Real symplectic form.
    """
    return np.block([[np.zeros((n,n)), np.eye(n)],\
                     [-np.eye(n), np.zeros((n,n))]])

def test_r(R):
    """
    Test if an matrix is real symplectic.
    """
    n = int(len(R)/2)
    WR = omega_r(n)
    return np.allclose(R @ WR @ R.T, WR)

def operator_real_symplectic(O, expm=True, theta=1):
    """
    Converts a first quantized operator into a real symplectic matrix.
    """
    n = O.shape[0]
    Op = O.full() if type(O) == qt.Qobj else O
    H, h = make_gaussian_operator(Op, np.zeros((n,n)), np.zeros(n))
    S, s = gaussian_complex_symplectic(H, h, expm=expm, theta=theta)
    R, r = complex_real_symplectic(S, s)
    return R, r

def symplectic_xyz():
    """
    Returns Pauli matrices expressed as real symplectic transformations.
    """
    return {"X": operator_real_symplectic(qt.sigmax()/2, expm=False)[0],\
            "Y": operator_real_symplectic(qt.sigmay()/2, expm=False)[0],\
            "Z": operator_real_symplectic(qt.sigmaz()/2, expm=False)[0]} 