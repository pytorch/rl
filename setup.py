#!/usr/bin/env python
# import distutils

import distutils.command.clean
import os
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

from build_tools import setup_helpers


def _get_pytorch_version():
    if 'PYTORCH_VERSION' in os.environ:
        return f"torch=={os.environ['PYTORCH_VERSION']}"
    return 'torch'


def _get_packages(branch_name, tag):
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
        for path in (ROOT_DIR / 'torchrl').glob('**/*.so'):
            print(f'removing \'{path}\'')
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / 'build',
        ]
        for path in build_dirs:
            if path.exists():
                print(f'removing \'{path}\' (and everything under it)')
                shutil.rmtree(str(path), ignore_errors=True)


def _run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR).decode('ascii').strip()
    except Exception:
        return None


pytorch_package_dep = _get_pytorch_version()
print('-- PyTorch dependency:', pytorch_package_dep)
branch = _run_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
tag = _run_cmd(['git', 'describe', '--tags', '--exact-match', '@'])

setup(
    name="torchrl",
    version="0.1",
    author="torchrl contributors",
    author_email="vmoens@fb.com",
    packages=_get_packages(branch, tag),
    # ext_modules=[CMakeExtension("torchrl", "./torchrl")],
    # cmdclass=dict(build_ext=CMakeBuild, test=GoogleTestCommand),
    ext_modules=setup_helpers.get_ext_modules(),
    cmdclass={
        'build_ext': setup_helpers.CMakeBuild,
        'clean': clean,
    },
    install_requires=[pytorch_package_dep],
)
