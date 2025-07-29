import os
import sys
import glob
import logging
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

def get_extensions():
    extension = CppExtension

    extra_link_args = []
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
    setup_kwargs = {
        "ext_modules": get_extensions(),
        "cmdclass": {"build_ext": BuildExtension.with_options()},
    }
    
    setup(**setup_kwargs)

if __name__ == "__main__":
    main()
