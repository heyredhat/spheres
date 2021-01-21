*******
spheres
*******

toolbox for higher spin and symmetrization

.. image:: https://readthedocs.org/projects/spheres/badge/?version=latest
	:target: https://spheres.readthedocs.io/en/latest/?badge=latest

.. image:: https://badge.fury.io/py/spheres.svg
	:target: https://badge.fury.io/py/spheres

.. image:: https://travis-ci.com/heyredhat/spheres.svg?branch=main
	:target: https://travis-ci.com/github/heyredhat/spheres

.. image:: https://codecov.io/gh/heyredhat/spheres/branch/main/graph/badge.svg?token=980CL7KIFL
	:target: https://codecov.io/gh/heyredhat/spheres

.. image:: https://pyup.io/repos/github/heyredhat/spheres/shield.svg
	:target: https://pyup.io/repos/github/heyredhat/spheres/

.. image:: https://pyup.io/repos/github/heyredhat/spheres/python-3-shield.svg
	:target: https://pyup.io/repos/github/heyredhat/spheres/


.. image:: https://raw.githubusercontent.com/heyredhat/spheres/main/stereographic_projection.jpg
   :align: center

`spheres` provides tools for dealing with higher spin systems and for symmetrized quantum circuits. Among other things, we provide implementations of:

1. The "Majorana stars" representation of a spin-j state as a degree-2j complex polynomial defined on the extended complex plane (the Riemann sphere). The eponymous stars are the 2j roots of this polynomial and each can be interpreted as a quantum of angular momentum contributing 1/2 in the specified direction. This polynomial can be defined in terms of the components of a ``|j, m>`` vector or in terms of a spin coherent wavefunction ``<-xyz|psi>``, where ``|psi>`` is the spin state and ``<-xyz|`` is the adjoint of a spin coherent state at the point antipodal to xyz. 

2. The "symmetrized spinors" representation of a spin-j state as 2j symmetrized spin-1/2 states (aka qubits). Indeed, the basis states of 2j symmetrized qubits are in 1-to-1 relation to the ``|j, m>`` basis states of a spin-j. Such a representation is naturally useful for simulating spin-j states on a qubit based quantum computer, and we provide circuits for preparing such states.

3. The "Schwinger oscillator" representation of a spin-j state as the total energy 2j subspace of two quantum harmonic oscillators. Indeed, the full space of the two oscillators furnishes a representation of spin with a variable j value: a superposition of j values. This construction can be interpreted as the "second quantization" of qubit, and is appropriate for implementation of photonic quantum computers.

Everything is accompanied by 3D visualizations thanks to `vpython` (and `matplotlib`) and interfaces to popular quantum computing libraries from `qutip` to `StrawberryFields`.

Finally, we provide tools for implementing a form of quantum error correction or "stablization" by harnessing the power of symmetrization. We provide automatic generation of circuits which perform a given quantum experiment multiple times in parallel while periodically projecting them all jointly into the symmetric subspace, which in principle increases the reliability of the computation under noisy conditions.

`spheres` is a work in progress! Beware!

Get started:

.. code-block:: 
	
	from spheres import *
	vsphere = MajoranaSphere(qt.rand_ket(3))
	vsphere.evolve(qt.rand_herm(3))

Check out the `documentation <https://spheres.readthedocs.io/>`_.

For more information and reference material, including jupyter notebooks, visit `heyredhat.github.io <https://heyredhat.github.io>`_.

Special thanks to the `Quantum Open Source Foundation <https://qosf.org/>`_.

