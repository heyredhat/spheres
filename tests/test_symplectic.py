import pytest
from spheres import *

def test_symplectic():
    H, h = random_gaussian_operator(4)
    S, s = gaussian_complex_symplectic(H, h)
    assert is_complex_symplectic(S)

    R1, r1 = complex_real_symplectic(S, s)
    R2, r2 = complex_real_symplectic2(S, s)
    assert np.allclose(R1, R2)
    assert np.allclose(r1, r2)
    assert is_real_symplectic(R1)

    assert is_real_symplectic(operator_real_symplectic(qt.rand_herm(4))[0])
