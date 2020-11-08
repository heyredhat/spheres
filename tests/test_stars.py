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
	sph = rand_sph()
	assert np.allclose(xyz_sph(sph_xyz(sph)), sph)

def test_c_sph():
	c = rand_c()
	assert np.isclose(sph_c(c_sph(c)), c)

def test_sph_c():
	sph = rand_sph(unit=True)
	assert np.allclose(c_sph(sph_c(sph)), sph)

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

def test_spinor_sph():
	spinor = qt.rand_ket(2)
	assert compare_nophase(sph_spinor(spinor_sph(spinor)), spinor)

def test_sph_spinor():
	sph = rand_sph(unit=True)
	assert np.allclose(spinor_sph(sph_spinor(sph)), sph)

def test_spin_poly():
	for i in range(5):
		spin = qt.basis(5, i)
		assert np.allclose(poly_spin(spin_poly(spin)), spin)

	j = 2
	spin = qt.rand_ket(int(2*j+1))

	assert np.allclose(poly_spin(spin_poly(spin)), spin)

	c_roots = spin_c(spin)
	spinor_roots = spin_spinors(spin)
	xyz_roots = spin_xyz(spin)
	spherical_roots = spin_sph(spin)

	projective = spin_poly(spin, projective=True)
	homogeneous = spin_poly(spin, homogeneous=True)
	cartesian = spin_poly(spin, cartesian=True)
	spherical = spin_poly(spin, spherical=True)

	for c in c_roots:
		assert np.isclose(projective(c), 0)
	for spinor in spinor_roots:
		assert np.isclose(homogeneous(spinor), 0)
	for xyz in xyz_roots:
		assert np.isclose(cartesian(xyz), 0)
	for sph in spherical_roots:
		assert np.isclose(spherical(sph), 0)

	random_c = rand_c()
	random_spinor = c_spinor(random_c)
	random_xyz = c_xyz(random_c)
	random_sph = c_sph(random_c)

	assert np.all(np.isclose(\
					np.array([cartesian(random_xyz),\
							  #homogeneous(random_spinor),\
					 		  spherical(random_sph)]), 
									projective(random_c)))

	normalized_projective = spin_poly(spin, projective=True, normalized=True)
	normalized_homogeneous = spin_poly(spin, homogeneous=True, normalized=True)
	normalized_cartesian = spin_poly(spin, cartesian=True, normalized=True)
	normalized_spherical = spin_poly(spin, spherical=True, normalized=True)

	assert np.all(np.isclose(\
					np.array([normalized_projective(random_c),\
							  #normalized_homogeneous(random_spinor),\
					 		  normalized_cartesian(random_xyz),\
					 		  normalized_spherical(random_sph)]), 
									spin_coherent(j, -random_xyz).dag()*spin))

def test_spin_xyz():
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

def test_spin_c():
	spin = qt.rand_ket(5)
	assert compare_nophase(c_spin(spin_c(spin)), spin)

def test_c_spin():
	c = [rand_c() for i in range(3)]
	assert compare_unordered(spin_c(c_spin(c)), c)

def test_spin_sph():
	spin = qt.rand_ket(5)
	assert compare_nophase(sph_spin(spin_sph(spin)), spin)

def test_sph_spin():
	sph = [rand_sph() for i in range(3)]
	assert compare_unordered(spin_sph(sph_spin(sph)), sph)

def test_poleflip():
	spin = qt.rand_ket(5)
	assert compare_nophase(poleflip(spin),\
				xyz_spin([c_xyz(c, pole="north")\
					for c in spin_c(spin)]))
	assert compare_nophase(poleflip(spin),\
				c_spin([poleflip(c) for c in spin_c(spin)]))

	poly = spin_poly(spin, projective=True, normalized=True)
	flipped_poly = spin_poly(poleflip(spin), projective=True, normalized=True)

	assert np.isclose(poly(0), flipped_poly(np.inf).conj())
	assert np.isclose(poly(np.inf), flipped_poly(0).conj()) # note

def test_antipodal():
	spin = qt.rand_ket(5)
	assert compare_nophase(antipodal(spin),\
				xyz_spin([-xyz for xyz in spin_xyz(spin)]))
	assert compare_nophase(antipodal(spin),\
				c_spin([antipodal(c) for c in spin_c(spin)]))

def test_mobius():
	pass
