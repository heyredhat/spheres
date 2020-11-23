from spheres import *

spin = qt.rand_ket(3)
m = MajoranaSphere(spin, position=vp.vector(0,3,0))
s = SchwingerSpheres()

s.raise_spin(spin)