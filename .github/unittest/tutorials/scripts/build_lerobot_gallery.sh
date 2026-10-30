#!/usr/bin/env bash

set -euo pipefail

# Use a separate checkout: editable TorchRL builds write their compiled extension
# into the source tree and must not replace the nightly docs build's extension.
root_dir="$(git rev-parse --show-toplevel)"
trial_dir="$(mktemp -d "${RUNNER_TEMP}/lerobot-gallery-XXXXXX")"
trial_repo="${trial_dir}/repo"
trial_python="${trial_dir}/env/bin/python"
git clone --quiet --shared --no-checkout "$root_dir" "$trial_repo"
git -C "$trial_repo" checkout --quiet --detach "$(git rev-parse HEAD)"
# The docs container may use an older Python than LeRobot supports.
python -m pip install uv
python -m uv venv --seed --python 3.12 "${trial_dir}/env"

# Match the gallery tooling used by the main docs build, while keeping the
# LeRobot environment on the PyTorch versions supported by LeRobot 0.6.1.
sphinx_version="$(python -c 'import sphinx; print(sphinx.__version__)')"
gallery_version="$(python -c 'import sphinx_gallery; print(sphinx_gallery.__version__)')"
cat > "${trial_dir}/constraints.txt" <<'CONSTRAINTS'
torch==2.11.0+cpu
torchvision==0.26.0+cpu
numpy>=2,<2.3
packaging>=24.2,<26
setuptools_scm>=8,<10
CONSTRAINTS
"$trial_python" -m pip install 'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' --index-url https://download.pytorch.org/whl/cpu
"$trial_python" -m pip install setuptools wheel 'setuptools_scm>=8,<10' 'packaging>=24.2,<26' ninja 'pybind11[global]' cmake
(
    cd "$trial_repo"
    "$trial_python" -m pip install --no-build-isolation -c "${trial_dir}/constraints.txt" -r docs/requirements.txt "sphinx==${sphinx_version}" "sphinx-gallery==${gallery_version}" -e '.[lerobot]'
    "$trial_python" -m pip check
    "$trial_python" -c 'import torch, torchvision; assert torch.__version__ == "2.11.0+cpu", torch.__version__; assert torchvision.__version__ == "0.26.0+cpu", torchvision.__version__; assert torch.version.cuda is None'
    cd docs
    # A dummy build still runs Sphinx Gallery, but avoids writing a second HTML site.
    PYOPENGL_PLATFORM=egl MUJOCO_GL=egl TORCHRL_TUTORIALS_FAST=1 "$trial_python" -m sphinx -b dummy -j 4 \
        -D 'sphinx_gallery_conf.filename_pattern=lerobot_export\.py$' \
        ./source "${trial_dir}/dummy-build"
)

# Copy the gallery page, checksum, notebook, and matching images into the main
# checkout. Sphinx Gallery will reuse the page during the nightly docs pass.
python - "$trial_repo/docs/source/tutorials" "$root_dir/docs/source/tutorials" <<'PY'
from pathlib import Path
from shutil import copy2
import sys

source, target = map(Path, sys.argv[1:])
page = source / "lerobot_export.rst"
assert page.is_file(), "LeRobot gallery page was not generated"
assert (source / "lerobot_export.py.md5").is_file(), "LeRobot gallery checksum is missing"
contents = page.read_text()
assert "<video" in contents and "data:video/mp4;base64," in contents, "LeRobot gallery page has no embedded MP4"
target.mkdir(parents=True, exist_ok=True)
for path in source.glob("lerobot_export.*"):
    copy2(path, target / path.name)
for path in (source / "images").rglob("sphx_glr_lerobot_export*"):
    if path.is_file():
        dest = target / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy2(path, dest)
print("Copied the executed LeRobot gallery page and embedded video")
PY
