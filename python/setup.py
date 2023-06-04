from setuptools import setup
from Cython.Build import cythonize

setup(
    name="bystro",
    package_dir={"bystro": "python/bystro"},
    ext_modules=cythonize(
        "python/bystro/search/**/*.pyx",
        build_dir="build",
        compiler_directives={"language_level": "3"},
    ),
    extras_require={
        ":platform.machine == 'x86_64' or platform.machine == 'AMD64' or platform.machine == 'aarch64'": [
            "isal"
        ]
    },
    zip_safe=False,
)
