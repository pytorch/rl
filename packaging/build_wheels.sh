#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=wheel
setup_env 0.1.0
setup_wheel_python
pip_install numpy pyyaml future ninja
pip_install --upgrade setuptools
setup_pip_pytorch_version
python setup.py clean

if [[ "$OSTYPE" == "msys" ]]; then
  echo "ERROR: Windows installation is not supported yet." && exit 100
else
    python setup.py bdist_wheel
    auditwheel repair dist/torchrl*.whl
    ls wheelhouse
fi
