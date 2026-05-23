from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        "classtrees.tree_module",
        sources=["src/classtrees/tree_module.c", "src/classtrees/tree.c"],
        include_dirs=[numpy.get_include(), "src/classtrees"],
        extra_compile_args=["-O3", "-DNDEBUG"]
    )
]

setup(
    name="classtrees",
    version="1.0.0",
    package_dir={"": "src"},
    packages=["classtrees"],
    ext_modules=ext_modules,
)
