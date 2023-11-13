# from distutils.core import setup
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['cnos'],
    package_dir={'': 'lib'}
)

setup(**setup_args)
