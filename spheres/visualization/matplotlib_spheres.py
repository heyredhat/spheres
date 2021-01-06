import pylab
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from ..stars import *

from IPython.display import HTML

def viz_spin(spin, show_arrows=True, show=True):
    """
    Visualizes a spin-j state with matplotlib.

    Parameters
    ----------
        spin : qt.Qobj

        show_arrows : bool
            If True, also shows vectors pointing to the stars.

        show : bool
            Whether to automatically display the figure.

    Returns
    -------

    """
    fig = pylab.figure()
    ax = Axes3D(fig)
    sphere = qt.Bloch(fig=fig, axes=ax)
    sphere.point_size=[300]*(spin.shape[0]-1)
    stars = spin_xyz(spin)
    sphere.add_points(stars.T)
    if show_arrows:
        sphere.add_vectors(stars)
    sphere.make_sphere()
    if show:
        pylab.show()
    return fig

def animate_spin(spin, H, dt=0.1, T=100, show_arrows=True, show=True, html_animation=False, filename=None, fps=20):
    """
    Visualizes the evolution of a spin-j state with matplotlib.

    .. code-block:: python

        %matplotlib notebook
        animate_spin(qt.rand_ket(3), qt.rand_herm(3))

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

        H : qt.Qobj
            Hamiltonian.

        dt : float
            Time step.

        T : float
            Time interval.

        show_arrows : bool
            If True, also shows vectors pointing to the stars.

        show : bool
            Whether to automatically display the figure.

        html_animation : bool
            Whether to return an HTML video.

        filename : str
            Where to save the resulting animation.

        fps : int
            Frames per second.

    Returns
    -------
        matplotlib.animation.FuncAnimation or IPython.core.display.HTML
    """
    U = (-1j*H*dt).expm()

    fig = pylab.figure()
    ax = Axes3D(fig)
    
    n = spin.shape[0]-1
    sphere = qt.Bloch(fig=fig, axes=ax)
    sphere.point_size=[300]*n
    sphere.make_sphere()
    
    history = [spin_xyz(spin)]
    for t in np.linspace(0, T, int(T/dt)):
        spin = U*spin
        history.append(spin_xyz(spin))

    def anim(i):
        nonlocal history, sphere
        sphere.clear()
        sphere.add_points(history[i].T)
        if show_arrows:
            sphere.add_vectors(history[i])
        sphere.make_sphere()
        return ax

    ani = animation.FuncAnimation(fig, anim, range(int(T/dt)), repeat=False)
    if filename:
        ani.save(filename, fps=fps)
    if show:
        pylab.show()
    return HTML(ani.to_html5_video()) if html_animation else ani


