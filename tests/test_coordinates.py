import pytest
from spheres import *

def test_c_xyz():
    c = rand_c()
    assert np.isclose(xyz_c(c_xyz(c)), c)

    C = rand_c(n=5)
    assert np.allclose(xyz_c(c_xyz(C)), C, equal_nan=True)

def test_xyz_c():
    xyz = rand_xyz()
    assert np.allclose(c_xyz(xyz_c(xyz)), xyz)

    XYZ = rand_xyz(n=5)
    assert np.allclose(c_xyz(xyz_c(XYZ)), XYZ)

def test_xyz_sph():
    xyz = rand_xyz()
    assert np.allclose(sph_xyz(xyz_sph(xyz)), xyz)

    XYZ = rand_xyz(n=5)
    assert np.allclose(sph_xyz(xyz_sph(XYZ)), XYZ)

def test_sph_xyz():
    sph = rand_sph()
    assert np.allclose(xyz_sph(sph_xyz(sph)), sph)

    SPH = rand_sph(n=5)
    assert np.allclose(xyz_sph(sph_xyz(SPH)), SPH)

def test_c_sph():
    c = rand_c()
    assert np.isclose(sph_c(c_sph(c)), c)

    C = rand_c(n=5)
    assert np.allclose(sph_c(c_sph(C)), C, equal_nan=True)

def test_sph_c():
    sph = rand_sph()
    assert np.allclose(c_sph(sph_c(sph)), sph)

    SPH = rand_sph(n=5)
    assert np.allclose(c_sph(sph_c(SPH)), SPH)

def test_c_spinor():
    c = rand_c()
    assert np.isclose(spinor_c(c_spinor(c)), c)

    C = rand_c(n=5)
    assert np.allclose(spinor_c(c_spinor(C)), C, equal_nan=True)

def test_spinor_c():
    spinor = qt.rand_ket(2)
    assert compare_nophase(c_spinor(spinor_c(spinor)), spinor)

    SPINOR = [qt.rand_ket(2) for i in range(5)]
    SPINOR2 = c_spinor(spinor_c(SPINOR))
    for i in range(5):
        assert compare_nophase(SPINOR[i], SPINOR2[i])

def test_xyz_spinor():
    xyz = rand_xyz()
    assert np.allclose(spinor_xyz(xyz_spinor(xyz)), xyz)

    XYZ = rand_xyz(n=5)
    assert np.allclose(spinor_xyz(xyz_spinor(XYZ)), XYZ)

def test_spinor_xyz():
    spinor = qt.rand_ket(2)
    assert compare_nophase(xyz_spinor(spinor_xyz(spinor)), spinor)

    SPINOR = [qt.rand_ket(2) for i in range(5)]
    SPINOR2 = xyz_spinor(spinor_xyz(SPINOR))
    for i in range(5):
        assert compare_nophase(SPINOR[i], SPINOR2[i])

def test_spinor_sph():
    spinor = qt.rand_ket(2)
    assert compare_nophase(sph_spinor(spinor_sph(spinor)), spinor)

    SPINOR = [qt.rand_ket(2) for i in range(5)]
    SPINOR2 = sph_spinor(spinor_sph(SPINOR))
    for i in range(5):
        assert compare_nophase(SPINOR[i], SPINOR2[i])

def test_sph_spinor():
    sph = rand_sph()
    assert np.allclose(spinor_sph(sph_spinor(sph)), sph)

    SPH = rand_sph(n=5)
    assert np.allclose(spinor_sph(sph_spinor(SPH)), SPH)