import pytest
from spheres.stars import *
from spheres.utils import *

def test_c_xyz():
	c = np.random.random() + 1j*np.random.random()
	assert np.isclose(xyz_c(c_xyz(c)), c)

def test_xyz_c():
	xyz = normalize(np.random.randn(3))
	assert np.isclose(c_xyz(xyz_c(xyz)), xyz).all()