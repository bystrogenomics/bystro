from setuptools import setup
from Cython.Build import cythonize
import glob
import os

# Search for .pyx files in the specified directories
pyx_files = glob.glob("python/bystro/**/**/*.pyx", recursive=True)

# Only cythonize if .pyx files are found
if pyx_files:
    ext_modules = cythonize(
        pyx_files,
        build_dir="build",
        compiler_directives={"language_level": "3"},
    )
else:
    ext_modules = []  # No extensions if no .pyx files are found

setup(
    name="bystro",
    ext_modules=ext_modules,
    # Other setup parameters here
)
