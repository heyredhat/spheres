import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spheres",
    version="0.3.0.5",
    author="Matthew Weiss",
    author_email="heyredhat@gmail.com",
    description="toolbox for higher spin and symmetrization",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/heyredhat/spheres",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ['numpy', 'qutip', 'matplotlib', 'vpython']
)