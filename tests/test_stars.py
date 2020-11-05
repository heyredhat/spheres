import pytest
from spheres import *

def test_c_xyz():
	c = rand_c()
	assert np.isclose(xyz_c(c_xyz(c)), c)

def test_xyz_c():
	xyz = rand_xyz()
	assert np.allclose(c_xyz(xyz_c(xyz)), xyz)

def test_xyz_sph():
	xyz = np.random.randn(3)
	assert np.allclose(sph_xyz(xyz_sph(xyz)), xyz)

def test_sph_xyz():
	sph = np.array([np.random.random(),\
					2*np.pi*np.random.random(),\
					np.pi*np.random.random()])
	assert np.allclose(xyz_sph(sph_xyz(sph)), sph)

def test_c_spinor():
	c = rand_c()
	assert np.isclose(spinor_c(c_spinor(c)), c)

def test_spinor_c():
	spinor = qt.rand_ket(2)
	assert compare_nophase(c_spinor(spinor_c(spinor)), spinor)

def test_xyz_spinor():
	xyz = normalize(np.random.randn(3))
	assert np.allclose(spinor_xyz(xyz_spinor(xyz)), xyz)

def test_spinor_xyz():
	spinor = qt.rand_ket(2)
	assert compare_nophase(xyz_spinor(spinor_xyz(spinor)), spinor)

def test_spin_xyz():
	for i in range(5):
		spin = qt.basis(5, i)
		assert np.allclose(poly_spin(spin_poly(spin)), spin)
	spin = qt.rand_ket(5)
	assert compare_nophase(xyz_spin(spin_xyz(spin)), spin)

def test_xyz_spin():
	xyz = rand_xyz(4)
	assert compare_unordered(spin_xyz(xyz_spin(xyz)), xyz)

def test_spin_spinors():
	spin = qt.rand_ket(5)
	assert compare_nophase(spinors_spin(spin_spinors(spin)), spin)

def test_spinors_spin():
	spinors = [qt.rand_ket(2) for i in range(4)]
	assert compare_spinors(spin_spinors(spinors_spin(spinors)), spinors)


