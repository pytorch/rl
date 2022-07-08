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
    env_path=$(dirname $bin_path)
    if [[ "$(uname)" == Darwin ]]; then
        # Install delocate to relocate the required binaries
        pip_install "delocate>=0.9"
    else
        cp "$bin_path/Library/bin/libpng16.dll" torchvision
        cp "$bin_path/Library/bin/libjpeg.dll" torchvision
    fi
else
    # Install auditwheel to get some inspection utilities
    pip_install auditwheel

    # Point to custom libraries
    export LD_LIBRARY_PATH=$(pwd)/ext_libraries/lib:$LD_LIBRARY_PATH
fi

if [[ "$OSTYPE" == "msys" ]]; then
  echo "ERROR: Windows installation is not supported yet." && exit 100
else
    python setup.py bdist_wheel
    if [[ "$(uname)" != Darwin ]]; then
      rename "linux_x86_64" "manylinux1_x86_64" dist/*.whl
    fi
fi

#if [[ "$(uname)" == Darwin ]]; then
#    pushd dist/
#    python_exec="$(which python)"
#    bin_path=$(dirname $python_exec)
#    env_path=$(dirname $bin_path)
#    for whl in *.whl; do
#        DYLD_FALLBACK_LIBRARY_PATH="$env_path/lib/:$DYLD_FALLBACK_LIBRARY_PATH" delocate-wheel -v --ignore-missing-dependencies $whl
#    done
#else
#    if [[ "$OSTYPE" == "msys" ]]; then
#        "$script_dir/windows/internal/vc_env_helper.bat" python $script_dir/wheel/relocate.py
#    else
#        LD_LIBRARY_PATH="/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH" python $script_dir/wheel/relocate.py
#    fi
#fi
