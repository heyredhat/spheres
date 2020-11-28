from spheres import *
from spheres.beams import *
from spheres.polyhedra import *

dt = 0.075
T = 2*np.pi
n_samples = 250

for j in np.arange(0, 6, 0.5):
    print("j : %f..." % j)
    d = int(2*j+1)
    if j == 0:
        filename = "beam_vids/beam_%d.mp4" % (d)
        animate_spin_beam(qt.rand_ket(1),\
                          qt.identity(1),\
                          dt=dt, T=T, n_samples=n_samples, filename=filename)
    else:
        for rotation in ['I', 'x', 'y', 'z']:
            for basis_state in ['x', 'y', 'z']:
                for i in range(d):
                    filename = "beam_vids/beam_%seig%d%d_%srot.mp4" % (basis_state, d, i, rotation)
                    animate_spin_beam(basis(d, i),\
                                    qt.jmat(j, rotation) if rotation != 'I' else qt.identity(d),\
                                    dt=dt, T=T, n_samples=n_samples, filename=filename)
            filename = "beam_vids/rand_beam_%d_%srot.mp4" % (d, rotation)
            animate_spin_beam(qt.rand_ket(d),\
                            qt.jmat(j, rotation) if rotation != 'I' else qt.identity(d),\
                            dt=dt, T=T, n_samples=n_samples, filename=filename)
            filename = "beam_vids/poly_beam_%d_%srot.mp4" % (d, rotation)
            animate_spin_beam(xyz_spin(equidistribute_points(int(2*j))),\
                            qt.jmat(j, rotation) if rotation != 'I' else qt.identity(d),\
                            dt=dt, T=T, n_samples=n_samples, filename=filename)
        filename = "beam_vids/rand_beam_%d_randH.mp4" % (d)
        animate_spin_beam(qt.rand_ket(d),\
                        qt.rand_herm(d),\
                        dt=dt, T=T, n_samples=n_samples, filename=filename)
