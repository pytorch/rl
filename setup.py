# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import distutils.command.clean
import glob
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import List

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

cwd = os.path.dirname(os.path.abspath(__file__))
try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    sha = "Unknown"


def get_version():
    version_txt = os.path.join(cwd, "version.txt")
    with open(version_txt, "r") as f:
        version = f.readline().strip()
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]
    return version


ROOT_DIR = Path(__file__).parent.resolve()


package_name = "torchrl"


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrl setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="torchrl",
        help="the name of this output wheel",
    )
    return parser.parse_known_args(argv)


def write_version_file(version):
    version_path = os.path.join(cwd, "torchrl", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


def _get_pytorch_version():
    # if "PYTORCH_VERSION" in os.environ:
    #     return f"torch=={os.environ['PYTORCH_VERSION']}"
    return "torch"


def _get_packages():
    exclude = [
        "build*",
        "test*",
        "torchrl.csrc*",
        "third_party*",
        "tools*",
    ]
    return find_packages(exclude=exclude)


ROOT_DIR = Path(__file__).parent.resolve()


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchrl extension
        for path in (ROOT_DIR / "torchrl").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


# def _run_cmd(cmd):
#     try:
#         return subprocess.check_output(cmd, cwd=ROOT_DIR).decode("ascii").strip()
#     except Exception:
#         return None


def get_extensions():
    extension = CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++14",
            "-fdiagnostics-color=always",
        ]
    }
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")
        extra_compile_args = {
            "cxx": [
                "-O0",
                "-fno-inline",
                "-g",
                "-std=c++14",
                "-fdiagnostics-color=always",
            ]
        }
        extra_link_args = ["-O0", "-g"]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchrl", "csrc")

    extension_sources = {
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    }
    sources = list(extension_sources)

    ext_modules = [
        extension(
            "torchrl._torchrl",
            sources,
            include_dirs=[this_dir],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


def _main(argv):
    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = "nightly" in name

    if is_nightly:
        version = get_nightly_version()
        write_version_file(version)
        print("Building wheel {}-{}".format(package_name, version))
        print(f"BUILD_VERSION is {os.getenv('BUILD_VERSION')}")
    else:
        version = get_version()

    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text()
    sys.argv = [sys.argv[0]] + unknown

    setup(
        # Metadata
        name=name,
        version=version,
        author="torchrl contributors",
        author_email="vmoens@fb.com",
        url="https://github.com/pytorch/rl",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD",
        # Package info
        packages=find_packages(exclude=("test", "tutorials")),
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
        install_requires=[
            pytorch_package_dep,
            "numpy",
            "packaging",
            "cloudpickle",
            "tensordict>=0.1.0",
        ],
        extras_require={
            "atari": [
                "gym<=0.24",
                "atari-py",
                "ale-py",
                "gym[accept-rom-license]",
                "pygame",
            ],
            "dm_control": ["dm_control"],
            "gym_continuous": ["mujoco-py", "mujoco"],
            "rendering": ["moviepy"],
            "tests": ["pytest", "pyyaml", "pytest-instafail", "scipy"],
            "utils": [
                "tensorboard",
                "wandb",
                "tqdm",
                "hydra-core>=1.1",
                "hydra-submitit-launcher",
            ],
            "checkpointing": [
                "torchsnapshot",
            ],
        },
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":

    _main(sys.argv[1:])
