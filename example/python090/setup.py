from setuptools import setup
from setuptools.extension import Extension

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        name="core",
        sources=["core.pyx"],
        language="c++",
        libraries=["core"],
    )
]

setup(
    name="core",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules),
    include_dirs=[
        numpy.get_include(),
    ],
)
