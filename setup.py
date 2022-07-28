# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import distutils.command.clean
import glob
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

cwd = os.path.dirname(os.path.abspath(__file__))
version_txt = os.path.join(cwd, "version.txt")
with open(version_txt, "r") as f:
    version = f.readline().strip()


ROOT_DIR = Path(__file__).parent.resolve()


try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    sha = "Unknown"
package_name = "torchrl"

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]


def write_version_file():
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

    extension_sources = set(
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    )
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


def _main():
    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text()

    setup(
        # Metadata
        name="torchrl",
        version=version,
        author="torchrl contributors",
        author_email="vmoens@fb.com",
        url="https://github.com/facebookresearch/rl",
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
        install_requires=[pytorch_package_dep, "numpy", "packaging"],
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
            "tests": ["pytest", "pyyaml", "pytest-instafail"],
            "utils": [
                "tensorboard",
                "wandb",
                "tqdm",
                "hydra-core>=1.1",
                "hydra-submitit-launcher",
            ],
        },
        zip_safe=False,
        # classifiers = [
        #    "Programming Language :: Python :: 3",
        #    "License :: OSI Approved :: MIT License",
        #    "Operating System :: OS Independent",
        # ]
    )


if __name__ == "__main__":

    write_version_file()
    print("Building wheel {}-{}".format(package_name, version))
    print(f"BUILD_VERSION is {os.getenv('BUILD_VERSION')}")
    _main()
