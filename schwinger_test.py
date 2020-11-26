from spheres import *
from spheres.visualization import *

spin = qt.rand_ket(3)
m = MajoranaSphere(spin, position=vp.vector(0,3,0))
s = SchwingerSpheres(max_ex=5, show_plane=True)
#s.raise_spin(spin)

H = s.randomH()