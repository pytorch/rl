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
fi

if [[ "$(uname)" == Darwin ]]; then
    pushd dist/
    python_exec="$(which python)"
    bin_path=$(dirname $python_exec)
    env_path=$(dirname $bin_path)
    for whl in *.whl; do
        DYLD_FALLBACK_LIBRARY_PATH="$env_path/lib/:$DYLD_FALLBACK_LIBRARY_PATH" delocate-wheel -v --ignore-missing-dependencies $whl
    done
else
    if [[ "$OSTYPE" == "msys" ]]; then
        "$script_dir/windows/internal/vc_env_helper.bat" python $script_dir/wheel/relocate.py
    else
        LD_LIBRARY_PATH="/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH" python $script_dir/wheel/relocate.py
    fi
fi
