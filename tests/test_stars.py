import pytest
from spheres import *

def test_c_xyz():
	c = rand_c()
	assert np.isclose(xyz_c(c_xyz(c)), c)

def test_xyz_c():
	xyz = normalize(np.random.randn(3))
	assert np.isclose(c_xyz(xyz_c(xyz)), xyz).all()

def test_xyz_sph():
	xyz = np.random.randn(3)
	assert np.isclose(sph_xyz(xyz_sph(xyz)), xyz).all()

def test_sph_xyz():
	sph = np.array([np.random.random(),\
					2*np.pi*np.random.random(),\
					np.pi*np.random.random()])
	assert np.isclose(xyz_sph(sph_xyz(sph)), sph).all()

def test_c_spinor():
	c = rand_c()
	assert np.isclose(spinor_c(c_spinor(c)), c)

def test_spinor_c():
	spinor = qt.rand_ket(2)
	assert np.isclose(c_spinor(spinor_c(spinor)), normalize_phase(spinor)).all()

def test_xyz_spinor():
	xyz = normalize(np.random.randn(3))
	assert np.isclose(spinor_xyz(xyz_spinor(xyz)), xyz).all()

def test_spinor_xyz():
	spinor = qt.rand_ket(2)
	assert np.isclose(xyz_spinor(spinor_xyz(spinor)), normalize_phase(spinor)).all()