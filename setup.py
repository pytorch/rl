# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import distutils.command.clean
import os
import shutil
import subprocess
from pathlib import Path

from build_tools import setup_helpers
from setuptools import setup, find_packages

cwd = os.path.dirname(os.path.abspath(__file__))
version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt, 'r') as f:
    version = f.readline().strip()

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    sha = 'Unknown'
package_name = 'torchrl'

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]

def write_version_file():
    version_path = os.path.join(cwd, 'torchrl', 'version.py')
    with open(version_path, 'w') as f:
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


def _run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR).decode("ascii").strip()
    except Exception:
        return None


def _main():
    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    setup(
        name="torchrl",
        version="0.1",
        author="torchrl contributors",
        author_email="vmoens@fb.com",
        packages=_get_packages(),
        ext_modules=setup_helpers.get_ext_modules(),
        cmdclass={
            "build_ext": setup_helpers.CMakeBuild,
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
    )


if __name__ == "__main__":
    print("Building wheel {}-{}".format(package_name, version))
    _main()


if __name__ == '__main__':
    write_version_file()
    setup(
        # Metadata
        name=package_name,
        version=version,
        author='PyTorch Core Team',
        url="https://github.com/pytorch/functorch",
        description='JAX-like composable function transforms for PyTorch',
        license='BSD',

        # Package info
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras,
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension_.with_options(no_python_abi_suffix=True),
            'clean': clean,
        })
