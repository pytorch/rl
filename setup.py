# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import os
import shutil
import subprocess
from pathlib import Path

from build_tools import setup_helpers as setup_h
from setuptools import setup, find_packages

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
    if "PYTORCH_VERSION" in os.environ:
        return f"torch=={os.environ['PYTORCH_VERSION']}"
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


def _main():
    from build_tools.setup_helpers import CMakeBuild
    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    setup(
        name="torchrl",
        version="0.1.0b",
        author="torchrl contributors",
        author_email="vmoens@fb.com",
        packages=_get_packages(),
        ext_modules=setup_h.get_ext_modules(),
        cmdclass={
            "build_ext": CMakeBuild,
            "clean": clean,
        },
        install_requires=[pytorch_package_dep, "numpy", "tensorboard", "packaging"],
        extras_require={
            "atari": ["gym", "atari-py", "ale-py", "gym[accept-rom-license]", "pygame"],
            "dm_control": ["dm_control"],
            "gym_continuous": ["mujoco-py", "mujoco"],
            "rendering": ["moviepy"],
            "tests": ["pytest", "pyyaml"],
            "utils": [
                "tqdm",
                "configargparse",
                "hydra-core>=1.1",
                "hydra-submitit-launcher",
            ],
        },
        url="https://github.com/facebookresearch/rl",
        # classifiers = [
        #    "Programming Language :: Python :: 3",
        #    "License :: OSI Approved :: MIT License",
        #    "Operating System :: OS Independent",
        # ]
    )


if __name__ == "__main__":

    write_version_file()
    print("Building wheel {}-{}".format(package_name, version))
    _main()
