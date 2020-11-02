import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spheres-heyredhat",
    version="0.0.1",
    author="Matthew Weiss",
    author_email="heyredhat@gmail.com",
    description="toolbox for higher spin and symmetrization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/spheres",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)