from spheres import *

j = 2
spin = qt.rand_ket(int(2*j+1))
stars = spin_xyz(spin)
spinors = spin_spinors(spin)
c = spin_c(spin)
s = spin_sph(spin)

poly = spin_poly(spin, projective=True, normalized=True)
hom = spin_poly(spin, homogeneous=True, normalized=True)
cart = spin_poly(spin, cartesian=True, normalized=True)
sphr = spin_poly(spin, spherical=True, normalized=True)

m = 1j
n = c_spinor(m)
o = c_xyz(m)
p = xyz_sph(o)

coh = lambda xyz: antipodal(spin).dag()*spin_coherent(j, xyz)
coh2 = lambda xyz: spin_coherent(j, -xyz).dag()*spin
