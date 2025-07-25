# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

from setuptools import Command, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

cwd = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(__file__).parent.resolve()


def write_version_file(version):
    version_path = os.path.join(cwd, "torchrl", "version.py")
    logging.info(f"Writing version file to: {version_path}")
    logging.info(f"Version to write: {version}")

    # Get PyTorch version during build
    try:
        import torch

        pytorch_version = torch.__version__
    except ImportError:
        pytorch_version = "unknown"

    # Get git sha
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
    except Exception:
        sha = "Unknown"

    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write(f"pytorch_version = '{pytorch_version}'\n")

    logging.info("Version file written successfully")


class clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove torchrl extension
        for path in (ROOT_DIR / "torchrl").glob("**/*.so"):
            logging.info(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                logging.info(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


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


def _main():
    # Always use "torchrl" as the project name for GitHub discovery
    # The version will be read from pyproject.toml

    # Handle nightly builds
    is_nightly = (
        any("nightly" in arg for arg in sys.argv) or os.getenv("TORCHRL_NIGHTLY") == "1"
    )
    logging.info(f"is_nightly: {is_nightly}")

    # Read version from version.txt
    version_txt = os.path.join(cwd, "version.txt")
    with open(version_txt) as f:
        base_version = f.readline().strip()

    if os.getenv("TORCHRL_BUILD_VERSION"):
        version = os.getenv("TORCHRL_BUILD_VERSION")
    elif is_nightly:
        from datetime import date

        today = date.today()
        version = f"{today.year}.{today.month}.{today.day}"
        logging.info(f"Using nightly version: {version}")
        # Update version.txt for nightly builds
        with open(version_txt, "w") as f:
            f.write(f"{version}\n")
    else:
        # For regular builds, append git hash for development versions
        try:
            import subprocess

            git_sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode("ascii")
                .strip()[:7]
            )
            version = f"{base_version}+{git_sha}"
            logging.info(f"Using development version: {version}")
        except Exception:
            version = base_version
            logging.info(f"Using base version: {version}")

    # Always write the version file to ensure it's up to date
    write_version_file(version)
    logging.info(f"Building torchrl-{version}")

    # Verify the version file was written correctly
    try:
        with open(os.path.join(cwd, "torchrl", "version.py")) as f:
            content = f.read()
            if f"__version__ = '{version}'" in content:
                logging.info(f"Version file correctly contains: {version}")
            else:
                logging.error(
                    f"Version file does not contain expected version: {version}"
                )
    except Exception as e:
        logging.error(f"Failed to verify version file: {e}")

    # Handle package name for nightly builds
    if is_nightly:
        package_name = "torchrl-nightly"  # Use torchrl-nightly for PyPI uploads
    else:
        package_name = "torchrl"  # Use torchrl for regular builds and GitHub discovery

    setup_kwargs = {
        "name": package_name,
        # Only C++ extension configuration
        "ext_modules": get_extensions(),
        "cmdclass": {
            "build_ext": BuildExtension.with_options(),
            "clean": clean,
        },
        "zip_safe": False,
        "package_data": {
            "torchrl": ["version.py"],
        },
        "include_package_data": True,
        "packages": ["torchrl"],
    }

    # Handle nightly tensordict dependency override
    if is_nightly:
        setup_kwargs["install_requires"] = [
            "torch>=2.1.0",
            "numpy",
            "packaging",
            "cloudpickle",
            "tensordict-nightly",
        ]

    # Override pyproject.toml settings for nightly builds
    if is_nightly:
        # Add all the metadata from pyproject.toml but override the name
        setup_kwargs.update(
            {
                "description": "A modular, primitive-first, python-first PyTorch library for Reinforcement Learning",
                "long_description": (Path(__file__).parent / "README.md").read_text(
                    encoding="utf8"
                ),
                "long_description_content_type": "text/markdown",
                "author": "torchrl contributors",
                "author_email": "vmoens@fb.com",
                "url": "https://github.com/pytorch/rl",
                "classifiers": [
                    "Programming Language :: Python :: 3.9",
                    "Programming Language :: Python :: 3.10",
                    "Programming Language :: Python :: 3.11",
                    "Programming Language :: Python :: 3.12",
                    "Operating System :: OS Independent",
                    "Development Status :: 4 - Beta",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "Topic :: Scientific/Engineering :: Artificial Intelligence",
                ],
            }
        )

    setup(**setup_kwargs)


if __name__ == "__main__":
    _main()
