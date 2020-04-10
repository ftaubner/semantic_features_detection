#cython: language_level=3

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("mask_tools.pyx"),
    include_dirs=[numpy.get_include()]
)

# Build with "python setup.py build_ext --inplace"
