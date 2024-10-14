from setuptools import setup  # type: ignore
from Cython.Build import cythonize  # type: ignore
import glob

# Search for all .pyx files recursively in the specified directory
pyx_files = glob.glob("python/bystro/**/*.pyx", recursive=True)

# Only cythonize if there are any .pyx files found
ext_modules = cythonize(
    pyx_files,
    build_dir="build",
    compiler_directives={"language_level": "3"},
) if pyx_files else []

setup(
    name="bystro",
    package_dir={"bystro": "python/bystro"},
    ext_modules=ext_modules,
    zip_safe=False,
)
