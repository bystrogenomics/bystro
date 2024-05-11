from setuptools import setup # type: ignore
from Cython.Build import cythonize # type: ignore

setup(
    name="bystro",
    package_dir={"bystro": "python/bystro"},
    ext_modules=cythonize(
        "python/bystro/search/**/*.pyx",
        build_dir="build",
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False
)
