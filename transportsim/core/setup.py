"""
setup.py  –  Build TransportSim C++ extension via pybind11.
"""

from setuptools import setup, Extension
import pybind11
import sys, os

ext = Extension(
    "_transportsim_core",
    sources=["bindings.cpp"],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(os.path.dirname(__file__), "include"),
    ],
    language="c++",
    extra_compile_args=[
        "-O3",
        "-std=c++17",
        "-ffast-math",
        "-march=native",
        "-fvisibility=hidden",
    ],
    extra_link_args=["-Wl,-strip-all"] if sys.platform != "darwin" else [],
)

setup(
    name="transportsim",
    version="1.0.0",
    description="TransportSim: C++ pipeline hydraulics for reactor networks",
    ext_modules=[ext],
    zip_safe=False,
)
