# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import distutils.sysconfig
import os
import platform
import subprocess
from pathlib import Path
from subprocess import CalledProcessError, check_output, STDOUT

import torch
from setuptools import Extension
from setuptools.command.build_ext import build_ext

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()
_TORCHRL_DIR = _ROOT_DIR / "torchrl"


def _get_build(var, default=False):
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        print(
            f"WARNING: Unexpected environment variable value `{var}={val}`. "
            f"Expected one of {trues + falses}"
        )
    return False


_BUILD_SOX = False if platform.system() == "Windows" else _get_build("BUILD_SOX", True)
_BUILD_KALDI = (
    False if platform.system() == "Windows" else _get_build("BUILD_KALDI", True)
)
_BUILD_RNNT = _get_build("BUILD_RNNT", True)
_USE_ROCM = False  # _get_build(
# "USE_ROCM", torch.cuda.is_available() and torch.version.hip is not None
# )
_USE_CUDA = False  # _get_build(
# "USE_CUDA", torch.cuda.is_available() and torch.version.hip is None
# )
_USE_OPENMP = (
    _get_build("USE_OPENMP", True)
    and "ATen parallel backend: OpenMP" in torch.__config__.parallel_info()
)
_TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", None)


def get_ext_modules():
    return [
        Extension(name="torchrl._torchrl", sources=[]),
    ]


# Based off of
# https://github.com/pybind/cmake_example/blob/580c5fd29d4651db99d8874714b07c0c49a53f8a/setup.py
class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        # Since two library files (libtorchrl and _torchrl) need to be
        # recognized by setuptools, we instantiate `Extension` twice. (see `get_ext_modules`)
        # This leads to the situation where this `build_extension` method is called twice.
        # However, the following `cmake` command will build all of them at the same time,
        # so, we do not need to perform `cmake` twice.
        # Therefore we call `cmake` only for `torchrl._torchrl`.
        if ext.name != "torchrl._torchrl":
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            "-DBUILD_TORCHRL_PYTHON_EXTENSION:BOOL=ON",
            f"-DUSE_CUDA:BOOL={'ON' if _USE_CUDA else 'OFF'}",
        ]
        build_args = ["--target", "install"]
        # Pass CUDA architecture to cmake
        if _TORCH_CUDA_ARCH_LIST is not None:
            # Convert MAJOR.MINOR[+PTX] list to new style one
            # defined at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
            _arches = _TORCH_CUDA_ARCH_LIST.replace(".", "").split(";")
            _arches = [
                arch[:-4] if arch.endswith("+PTX") else f"{arch}-real"
                for arch in _arches
            ]
            cmake_args += [f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(_arches)}"]

        # Default to Ninja
        if "CMAKE_GENERATOR" not in os.environ or platform.system() == "Windows":
            cmake_args += ["-GNinja"]
        if platform.system() == "Windows":
            import sys

            python_version = sys.version_info
            cmake_args += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                f"-DPYTHON_VERSION={python_version.major}.{python_version.minor}",
            ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(" ".join(["cmake", str(_ROOT_DIR)] + cmake_args))
        try:
            check_output(
                ["cmake", str(_ROOT_DIR)] + cmake_args,
                cwd=self.build_temp,
                stderr=STDOUT,
            )
        except CalledProcessError as exc:
            print(exc.output)

        try:
            check_output(
                ["cmake", "--build", "."] + build_args,
                cwd=self.build_temp,
                stderr=STDOUT,
            )
        except CalledProcessError as exc:
            print(exc.output)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split(".")
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = ".".join(without_abi)
        return ext_filename
