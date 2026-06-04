from setuptools import setup, Extension
import numpy
import sys

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-DNDEBUG", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "classtrees._tree_module",
        sources=["src/classtrees/_tree_module.c",
                 "src/classtrees/tree.c",
                 "src/classtrees/tree_bootstrap.c",
                 "src/classtrees/randomforest.c",
                 "src/classtrees/checks.c",],
        include_dirs=[numpy.get_include(), "src/classtrees"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="classtrees",
    package_dir={"": "src"},
    packages=["classtrees"],
    ext_modules=ext_modules,
)