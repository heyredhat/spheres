import pylab
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def animate_spin(state, H, dt=0.1, T=100, filename=None):
    """
    Animate Majorana stars with matplotlib.
    """
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

#from IPython.display import HTML
#HTML(ani.to_html5_video())

#import matplotlib.pyplot as plt

def viz_spin(spin):
    """
    Visualize Majorana stars with matplotlib.
    """
    fig = pylab.figure()
    ax = Axes3D(fig)
    sphere = qt.Bloch(fig=fig, axes=ax)
    sphere.point_size=[300]*(spin.shape[0]-1)
    stars = spin_xyz(spin)
    sphere.add_points(stars.T)
    sphere.add_vectors(stars)
    sphere.make_sphere()
    pylab.show()
    return sphere
