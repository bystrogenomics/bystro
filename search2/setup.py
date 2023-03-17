from setuptools import setup
from Cython.Build import cythonize

setup(
    name="search",
    package_dir={'search': 'python/search/'},
    ext_modules=cythonize("python/search/**/*.pyx", build_dir="build"),
    zip_safe=False,
)