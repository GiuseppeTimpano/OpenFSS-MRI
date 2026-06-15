"""
Build the 3D Felzenszwalb Cython extension in-place.

    pip install cython
    python setup.py build_ext --inplace
"""
import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "_felz3d",
        ["_felz3d.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="felz3d",
    ext_modules=cythonize(extensions, language_level="3"),
)
