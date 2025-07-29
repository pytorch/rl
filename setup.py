import glob
import logging
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

logger = logging.getLogger(__name__)


def get_extensions():
    """Build C++ extensions with platform-specific compiler flags.

    This function configures the C++ extension build process with appropriate
    compiler flags for different platforms:
    - Windows (MSVC): Uses /O2, /std:c++17, /EHsc flags
    - Unix-like (GCC/Clang): Uses -O3, -std=c++17, -fdiagnostics-color=always flags

    Returns:
        list: List of CppExtension objects to be built
    """
    extension = CppExtension
    extra_link_args = []

    # Platform-specific compiler flags
    if sys.platform == "win32":
        # MSVC flags for Windows
        extra_compile_args = {
            "cxx": [
                "/O2",  # Optimization level 2 (equivalent to -O3)
                "/std:c++17",  # C++17 standard
                "/EHsc",  # Exception handling model
            ]
        }
        debug_mode = os.getenv("DEBUG", "0") == "1"
        if debug_mode:
            logging.info("Compiling in debug mode")
            extra_compile_args = {
                "cxx": [
                    "/Od",  # No optimization (equivalent to -O0)
                    "/Zi",  # Generate debug info
                    "/std:c++17",  # C++17 standard
                    "/EHsc",  # Exception handling model
                ]
            }
            extra_link_args = ["/DEBUG"]
    else:
        # GCC/Clang flags for Unix-like systems
        extra_compile_args = {
            "cxx": [
                "-O3",
                "-std=c++17",
                "-fdiagnostics-color=always",
            ]
        }
        debug_mode = os.getenv("DEBUG", "0") == "1"
        if debug_mode:
            logging.info("Compiling in debug mode")
            extra_compile_args = {
                "cxx": [
                    "-O0",
                    "-fno-inline",
                    "-g",
                    "-std=c++17",
                    "-fdiagnostics-color=always",
                ]
            }
            extra_link_args = ["-O0", "-g"]

    extensions_dir = "torchrl/csrc"

    # Get just the filenames, not full paths
    cpp_files = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    sources = [os.path.relpath(f) for f in cpp_files]

    include_dirs = ["."]
    python_include_dir = os.getenv("PYTHON_INCLUDE_DIR")
    if python_include_dir is not None:
        include_dirs.append(python_include_dir)
    ext_modules = [
        extension(
            "torchrl._torchrl",
            sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


def main():
    """Main setup function for building TorchRL with C++ extensions."""
    setup_kwargs = {
        "ext_modules": get_extensions(),
        "cmdclass": {"build_ext": BuildExtension.with_options()},
        "packages": ["torchrl"],
        "package_data": {
            "torchrl": ["version.py"],
        },
        "include_package_data": True,
        "zip_safe": False,
    }

    setup(**setup_kwargs)


if __name__ == "__main__":
    main()
