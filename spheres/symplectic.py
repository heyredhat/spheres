"""
Functions for converting between Gaussian Hamiltonians, complex symplectic matrices, and real symplectic matrices.
"""
import numpy as np
import qutip as qt
import scipy as sc

def make_gaussian_operator(A, B=None, h=None):
    r"""
    Constructs Gaussian transformation in the form of a Hermitian matrix and a displacement vector.

    A Gaussian unitary can be written :math:`U = e^{iH}`, where :math:`H` is quadratic in creation and annihilation operators. 

    If we define :math:`\xi = \begin{pmatrix}\hat{a}_{0} \\ \hat{a}_{1} \\ \vdots \\ \hat{a}_{0}^{\dagger} \\ \hat{a}_{1}^{\dagger} \\ \vdots \end{pmatrix}` to be a vector of :math:`n` annihilation operators and :math:`n` creation operators, then :math:`H` can be written:

    .. math::
    
        H = \frac{1}{2}\xi^{\dagger}\textbf{H}\xi + \xi^{\dagger}\textbf{h}

    Where :math:`\textbf{H} = \begin{pmatrix} A & B \\ B^{*} & A^{*} \end{pmatrix}` is a :math:`2n \times 2n` Hermitian matrix and :math:`\textbf{h} = \begin{pmatrix} h \\ h^{*} \end{pmatrix}` is a :math:`2n` complex column vector. 
    
    :math:`A` is a Hermitian matrix and `B` is symmetric.
    
    Parameters
    ----------
        A : np.array
            Hermitian matrix.
        B : np.array
            Symmetric matrix.
        h : np.array
            Complex array.

    Returns
    -------
        H : np.array
            Gaussian operator.
        h : np.array
            Gaussian displacement.

    """
def make_gaussian_operator(A, B=None, h=None):
    B = np.zeros(A.shape) if type(B) == type(None) else B
    h = np.zeros(A.shape[0]) if type(h) == type(None) else h
    return np.block([[A, B],\
                     [B.conj(), A.conj()]]),\
           np.concatenate([h, h.conj()])

def random_gaussian_operator(n):
    r"""
    Returns random Gaussian transformation in the form of a Hermitian matrix and a displacement vector.
    
    Parameters
    ----------
        n : int
            The operator will be :math:`2n \times 2n` dimensions.
    
    Returns
    -------
        H : np.array
            Gaussian operator.
        h : np.array
            Gaussian displacement.
    
    """
    A = qt.rand_herm(n).full()
    B = np.random.randn(n, n) + 1j*np.random.randn(n, n) 
    B = B + B.T
    h = np.random.randn(n) + 1j*np.random.randn(n)
    return make_gaussian_operator(A, B, h)

def gaussian_complex_symplectic(H, h, expm=True, theta=2):
    r"""
    Converts a Gaussian transformation (in the form of a Hermitian matrix and a displacement vector)
    into a complex symplectic transformation (in the form of a complex symplectic matrix and displacement vector).

    Evolving all the creation and annihilation operators by a Gaussian unitary is equivalent to an affine transformation:

    .. math::
    
        e^{iH} \xi e^{-iH} = \textbf{S}\xi + \textbf{s}

    Where :math:`\textbf{S}` is a complex symplectic matrix: :math:`\textbf{S} = e^{-i\Omega_{c}\textbf{H}}` and :math:`\textbf{s} = (\textbf{S}-I_{2n})\textbf{H}^{-1}\textbf{h}`.

    Here the complex symplectic form is :math:`\Omega_{c} = \begin{pmatrix}I_{n} & 0 \\ 0 & -I_{n} \end{pmatrix}`, and if :math:`\textbf{H}` has no inverse, we take the pseudoinverse instead to calculate :math:`\textbf{h}`.
    
    Parameters
    ----------
        H : np.array
            Gaussian operator.
        h : np.array
            Gaussian displacement.
        expm : bool
            Whether to exponentiate.
        theta : float
            Exponentiation parameter.

    Returns
    -------
        S : np.array
            Complex symplectic matrix.
        s : np.array
            Complex symplectic displacement vector.
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
    r"""
    :math:`2n \times 2n` complex symplectic form: :math:`\Omega_{c} = \begin{pmatrix}I_{n} & 0 \\ 0 & -I_{n} \end{pmatrix}`.

    Parameters
    ----------
        n : int
            The dimension will be 2n.

    Returns
    -------
        W : np.array
            Complex symplectic form.
    """
    return sc.linalg.block_diag(np.eye(n), -np.eye(n))

def is_complex_symplectic(S):
    """
    Test if an matrix is complex symplectic.

    Parameters
    ----------
        S : np.array

    Returns
    -------
        is_symplectic : bool
    """
    n = int(len(S)/2)
    WC = omega_c(n)
    return np.allclose(S @ WC @ S.conj().T, WC)
    
def complex_real_symplectic(S, s):
    r"""
    Converts a complex symplectic transformation into a real symplectic transformation.
    
    We can use a complex symplectic matrix to perform a Gaussian unitary transformation on a vector of creation and annihilation operators. 
    At the same time, there is an equivalent real symplectic matrix :math:`\textbf{R}` and real displacement vector :math:`\textbf{r}` that implements the same transformation on a vector of position and momentum operators.

    .. math::
    
        \vec{V} \rightarrow \textbf{R}\vec{V} + \textbf{r}

    We can easily convert between :math:`\xi`, the vector of annihilation and creation operators, and :math:`\vec{V}`, the vector of positions and momenta, via:

    .. math::
    
        \vec{V} = L\xi

    Where :math:`L = \frac{1}{\sqrt{2}}\begin{pmatrix} I_{n} & I_{n} \\ -iI_{n} & iI_{n} \end{pmatrix}`. 

    This comes from the definition of position and momentum operators in terms of creation and annihilation operators, e.g.:

    .. math::
    
        \hat{Q} = \frac{1}{\sqrt{2}}(\hat{a} + \hat{a}^{\dagger})

        \hat{P} = -\frac{i}{\sqrt{2}}(\hat{a} - \hat{a}^{\dagger})

    Therefore we can turn our complex symplectic transformation into a real symplectic transformation via:

    .. math::

        \textbf{R} = L\textbf{S}L^{\dagger}

        \textbf{r} = L\textbf{s}

    If we've represented a Gaussian state in terms of its first and second moments, then the real sympectic transformations act on them!

    Parameters
    ----------
        S : np.array
            Complex symplectic matrix.
        s : np.array
            Complex symplectic displacement vector.
    
    Returns
    -------
        R : np.array
            Real symplectic matrix.
        r : np.array
            Real symplectic displacement vector.
    """
    n = int(len(s)/2)
    L = (1/np.sqrt(2))*np.block([[np.eye(n), np.eye(n)],\
                             [-1j*np.eye(n), 1j*np.eye(n)]])
    return (L @ S @ L.conj().T).real, (L @ s).real

def complex_real_symplectic2(S, s):
    r"""
    Converts a complex symplectic matrix/vector to a real symplectic matrix/vector.
    Alternative construction:

    If we write :math:`\textbf{S}` as :math:`\begin{pmatrix} E & F \\ F^{*} & E^{*} \end{pmatrix}`, and :math:`\textbf{s}` as :math:`\begin{pmatrix} s \\ s^{*} \end{pmatrix}` then equivalently:

    .. math::
        
        \textbf{R} = \begin{pmatrix}\Re(E+F) & -\Im(E-F) \\ \Im(E+F) & \Re(E-F) \end{pmatrix}
        
        \textbf{r} = \sqrt{2}\begin{pmatrix} \Re(s) \\ \Im(s) \end{pmatrix}

    Parameters
    ----------
        S : np.array
            Complex symplectic matrix.
        s : np.array
            Complex symplectic displacement vector.
    
    Returns
    -------
        R : np.array
            Real symplectic matrix.
        r : np.array
            Real symplectic displacement vector.

    """
    n = int(len(s)/2)
    E, F = S[0:n, 0:n], S[0:n, n:]
    return np.block([[(E+F).real, -(E-F).imag],\
                     [(E+F).imag, (E-F).real]]),\
           np.sqrt(2)*np.concatenate([s[0:n].real,\
                                      s[0:n].imag])

def omega_r(n):
    r"""
    :math:`2n \times 2n` real symplectic form: :math:`\Omega_{c} = \begin{pmatrix}0 & I_{n} \\ -I_{n} & 0 \end{pmatrix}`.

    Parameters
    ----------
        n : int
            The dimension will be 2n.

    Returns
    -------
        W : np.array
            Real symplectic form.
    """
    return np.block([[np.zeros((n,n)), np.eye(n)],\
                     [-np.eye(n), np.zeros((n,n))]])

def is_real_symplectic(R):
    """
    Test if an matrix is real symplectic.

    Parameters
    ----------
        S : np.array

    Returns
    -------
        is_symplectic : bool
    """
    n = int(len(R)/2)
    WR = omega_r(n)
    return np.allclose(R @ WR @ R.T, WR)

def operator_real_symplectic(O, expm=True, theta=2):
    """
    Converts a first quantized operator into a real symplectic matrix via: :meth:`make_gaussian_operator`, :meth:`gaussian_complex_symplectic`, :math:`complex_real_symplectic`.

    Parameters
    ----------
        O : qt.Qobj
            Operator
        expm : bool
            Whether to exponentiate.
        theta : float
            Parameter for exponentiation.

    Returns
    -------
        R : np.array
            Real symplectic matrix.
        r : np.array
            Real symplectic displacement vector.
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

    Returns
    -------
        XYZ : dict
            Associates "x", "y", "z" to the corresponding real symplectic matrices.

    """
    return {"x": operator_real_symplectic(qt.sigmax()/8, expm=False)[0],\
            "y": operator_real_symplectic(qt.sigmay()/8, expm=False)[0],\
            "z": operator_real_symplectic(qt.sigmaz()/8, expm=False)[0]} 

def upgrade_single_mode_operator(O, i, n_modes):
    """
    Upgrades a single mode real symplectic operator O to act on the i'th of n modes (where the latter are represented in terms of their first and second moments.)

    Parameters
    ----------
        O : np.array
            Single mode operator.
        i : int
            Which mode to act on.
        n_modes : int
            Of how many modes.

    Returns
    -------
        U : np.array
            Upgraded operator.

    """
    I = np.eye(2*n_modes)
    cols = np.zeros((2,2), dtype=np.intp)
    cols[0,:], cols[1,:] = i, i+n_modes
    rows = cols.T
    I[rows,cols] = O[:]
    return I

def upgrade_two_mode_operator(O, i, j, n_modes):
    """
    Upgrades a two mode real symplectic matrix to act on subsystems i and j of n modes,
    (where the modes are represented in terms of their first and second moments).

    Parameters
    ----------
        O : np.array
            Two mode operator.
        i : int
            First mode to act on.
        j : int
            Second mode to act on.
        n_modes : int
            Of how many modes.

    Returns
    -------
        U : np.array
            Upgraded operator.
    """
    I = np.zeros((2*n_modes, 2*n_modes))
    cols = np.zeros((4,4), dtype=np.intp)
    cols[0,:], cols[1,:], cols[2,:], cols[3,:]  = i, j, i+n_modes, j+n_modes
    rows = cols.T
    I[rows,cols] = O[:]
    return I