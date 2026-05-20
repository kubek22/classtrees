from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        "classtrees.tree",
        sources=["src/classtrees/tree.c"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="classtrees",
    version="1.0.0",
    package_dir={"": "src"},
    packages=["classtrees"],
    ext_modules=ext_modules,
)
