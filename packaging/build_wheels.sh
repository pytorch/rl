#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=wheel
setup_env
setup_wheel_python
pip_install numpy pyyaml future ninja
pip_install --upgrade setuptools
setup_pip_pytorch_version
python setup.py clean

# Copy binaries to be included in the wheel distribution
if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    python_exec="$(which python)"
    bin_path=$(dirname $python_exec)
    if [[ "$(uname)" == Darwin ]]; then
        # Install delocate to relocate the required binaries
        pip_install "delocate>=0.9"
    fi
else
    # Install auditwheel to get some inspection utilities
    pip_install auditwheel

    # Point to custom libraries
    export LD_LIBRARY_PATH=$(pwd)/ext_libraries/lib:$LD_LIBRARY_PATH
fi

if [[ "$OSTYPE" == "msys" ]]; then
    IS_WHEEL=1 "$script_dir/windows/internal/vc_env_helper.bat" python setup.py bdist_wheel
else
    python setup.py bdist_wheel
    if [[ "$(uname)" != Darwin ]]; then
      rename "linux_x86_64" "manylinux1_x86_64" dist/*.whl
    fi
fi
