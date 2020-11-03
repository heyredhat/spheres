import pytest
from spheres.stars import *

def test_c_xyz():
	c = np.random.random() + 1j*np.random.random()
	assert xyz_c(c_xyz(c)) == c