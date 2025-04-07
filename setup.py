# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import distutils.command.clean
import glob
import logging
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
    with open(version_txt) as f:
        version = f.readline().strip()
    if os.getenv("TORCHRL_BUILD_VERSION"):
        version = os.getenv("TORCHRL_BUILD_VERSION")
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
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


def _get_pytorch_version(is_nightly, is_local):
    # if "PYTORCH_VERSION" in os.environ:
    #     return f"torch=={os.environ['PYTORCH_VERSION']}"
    if is_nightly:
        return "torch>=2.7.0.dev"
    elif is_local:
        return "torch"
    return "torch>=2.6.0"


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

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchrl", "csrc")

    extension_sources = {
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    }
    sources = list(extension_sources)

    include_dirs = [this_dir]
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


def _main(argv):
    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = "nightly" in name
    if is_nightly:
        tensordict_dep = "tensordict-nightly"
    else:
        tensordict_dep = "tensordict>=0.7.0"

    if is_nightly:
        version = get_nightly_version()
        write_version_file(version)
    else:
        version = get_version()
        write_version_file(version)
    TORCHRL_BUILD_VERSION = os.getenv("TORCHRL_BUILD_VERSION")
    logging.info(f"Building wheel {package_name}-{version}")
    logging.info(f"TORCHRL_BUILD_VERSION is {TORCHRL_BUILD_VERSION}")

    is_local = TORCHRL_BUILD_VERSION is None
    pytorch_package_dep = _get_pytorch_version(is_nightly, is_local)
    logging.info("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text(encoding="utf8")
    sys.argv = [sys.argv[0]] + unknown

    extra_requires = {
        "atari": [
            "gym",
            "atari-py",
            "ale-py",
            "gym[accept-rom-license]",
            "pygame",
        ],
        "dm_control": ["dm_control"],
        "gym_continuous": ["gymnasium<1.0", "mujoco"],
        "rendering": ["moviepy<2.0.0"],
        "tests": [
            "pytest",
            "pyyaml",
            "pytest-instafail",
            "scipy",
            "pytest-mock",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-benchmark",
            "pytest-rerunfailures",
            "pytest-error-for-skips",
            "",
        ],
        "utils": [
            "tensorboard",
            "wandb",
            "tqdm",
            "hydra-core>=1.1",
            "hydra-submitit-launcher",
            "git",
        ],
        "checkpointing": [
            "torchsnapshot",
        ],
        "offline-data": [
            "huggingface_hub",  # for roboset
            "minari",
            "requests",
            "tqdm",
            "torchvision",
            "scikit-learn",
            "pandas",
            "h5py",
            "pillow",
        ],
        "marl": ["vmas>=1.2.10", "pettingzoo>=1.24.1", "dm-meltingpot"],
        "open_spiel": ["open_spiel>=1.5"],
    }
    extra_requires["all"] = set()
    for key in list(extra_requires.keys()):
        extra_requires["all"] = extra_requires["all"].union(extra_requires[key])
    extra_requires["all"] = sorted(extra_requires["all"])
    setup(
        # Metadata
        name=name,
        version=version,
        author="torchrl contributors",
        author_email="vmoens@fb.com",
        url="https://github.com/pytorch/rl",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="MIT",
        # Package info
        packages=find_packages(
            exclude=(
                "test",
                "tutorials",
                "docs",
                "examples",
                "knowledge_base",
                "packaging",
            )
        ),
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension.with_options(),
            "clean": clean,
        },
        install_requires=[
            pytorch_package_dep,
            "numpy",
            "packaging",
            "cloudpickle",
            tensordict_dep,
        ],
        extras_require=extra_requires,
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
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
