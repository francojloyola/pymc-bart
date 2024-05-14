from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


# python setup.py build_ext --inplace
setup(
    name='tree',
    ext_modules=cythonize([
        Extension("algorithms", ["algorithms.pyx"]),
    ]),
    package_dir={'pymc_bart': ''},
    include_dirs=[numpy.get_include()]
)
