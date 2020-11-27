%matplotlib widget

import pylab
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from spheres.stars import *

def animate_spin(state, H, dt=0.1, T=100, filename=None):
    U = (-1j*H*dt).expm()

    fig = pylab.figure()
    ax = Axes3D(fig)
    
    n = state.shape[0]-1
    sphere = qt.Bloch(fig=fig, axes=ax)
    sphere.point_size=[300]*n
    
    sphere.make_sphere()
    
    history = [spin_xyz(state)]
    for t in range(T):
        state = U*state
        history.append(spin_xyz(state))

    def anim(i):
        nonlocal history, sphere
        sphere.clear()
        sphere.add_points(history[i].T)
        sphere.add_vectors(history[i])
        sphere.make_sphere()
        return ax

    ani = animation.FuncAnimation(fig,\
                                  anim,\
                                  range(T),\
                                  repeat=False)
    if filename:
        ani.save(filename, fps=20)
    #pylab.show()
    return ani

spin = qt.rand_ket(3)
H = qt.jmat(1, 'x')
ani = animate_spin(spin, H)

#from IPython.display import HTML
#HTML(ani.to_html5_video())
