import pytest
from spheres import *

def test_spin_sym():
	j = 2
	spin = qt.rand_ket(int(2*j+1))
	spinors = spin_spinors(spin)
	sym = symmetrize(spinors)
	S = spin_sym(j)
	assert compare_nophase(S*spin, sym)