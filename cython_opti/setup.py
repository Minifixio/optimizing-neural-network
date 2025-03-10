from distutils.core import setup
from Cython.Build import cythonize
import numpy

files = ["gradient_cython.pyx", "predict_cython.pyx", "cost_function_cython.pyx"]

setup(
    ext_modules=cythonize(files),
    compiler_directives={'language_level': "3"},
    include_dirs=[numpy.get_include()]
)