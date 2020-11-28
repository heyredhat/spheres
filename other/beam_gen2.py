from spheres import *
from spheres.beams import *
from spheres.polyhedra import *

dt = 0.04
T = 5*np.pi
n_samples = 250

for i in range(10):
    for j in np.arange(1, 6, 0.5):
        print("j : %f..." % j)
        d = int(2*j+1)
        filename = "beam_vids/etc2/rand_beam_%d_randH%d.mp4" % (d, i)
        animate_spin_beam(qt.rand_ket(d),\
                        qt.rand_herm(d),\
                        dt=dt, T=T, n_samples=n_samples, filename=filename)
