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
    zip_safe=False,
    install_requires=[
        'Canopy @ git+https://github.com/bystrogenomics/canopy.git@f6d3031494a692db040492fa1dead5faf19fef5a'
    ]
)
