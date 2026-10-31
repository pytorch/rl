#!/usr/bin/env bash

set -euo pipefail

# Use a separate source tree: editable TorchRL builds write their compiled
# extension into the tree and must not replace the nightly docs extension. The
# CI checkout can be a GitHub archive without a .git directory.
root_dir="$(pwd)"
trial_dir="$(mktemp -d "${RUNNER_TEMP}/lerobot-gallery-XXXXXX")"
trial_repo="${trial_dir}/repo"
trial_python="${trial_dir}/env/bin/python"
gallery_source="${trial_dir}/gallery-source"
mkdir -p "$trial_repo"
tar -C "$root_dir" --exclude='./.git' --exclude='./.docs_mujoco_menagerie' \
    --exclude='./docs/source/tutorials' --exclude='./docs/_local_build' \
    --exclude='./build' --exclude='./_tmp' --exclude='__pycache__' \
    --exclude='torchrl/_torchrl*.so' -cf - . | tar -C "$trial_repo" -xf -
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
    "$trial_python" -m pip install --no-build-isolation -c "${trial_dir}/constraints.txt" \
        "sphinx==${sphinx_version}" "sphinx-gallery==${gallery_version}" \
        matplotlib ipython 'mujoco>=3.8.1,<3.9.0' -e '.[lerobot]'
    "$trial_python" -m pip check
    "$trial_python" -c 'import torch, torchvision; assert torch.__version__ == "2.11.0+cpu", torch.__version__; assert torchvision.__version__ == "0.26.0+cpu", torchvision.__version__; assert torch.version.cuda is None'
    # Build only this gallery page, avoiding the unrelated docs extensions and
    # autosummary work. Keep these paths and capture settings aligned with conf.py
    # so the generated Sphinx Gallery labels and output match the main build.
    mkdir -p "${gallery_source}/reference/generated/tutorials"
    cp tutorials/sphinx-tutorials/lerobot_export.py "${gallery_source}/reference/generated/tutorials/"
    cat > "${gallery_source}/reference/generated/tutorials/GALLERY_HEADER.rst" <<'GALLERY'
Tutorials
=========
GALLERY
    cat > "${gallery_source}/index.rst" <<'INDEX'
Tutorials
=========

.. toctree::

   tutorials/index
INDEX
    cat > "${gallery_source}/conf.py" <<'CONF'
project = "torchrl"
extensions = ["sphinx_gallery.gen_gallery"]
sphinx_gallery_conf = {
    "examples_dirs": "reference/generated/tutorials/",
    "gallery_dirs": "tutorials",
    "doc_module": ("torchrl",),
    "reference_url": {"torchrl": None},
    "filename_pattern": "lerobot_export\\.py$",
    "abort_on_example_error": True,
    "capture_repr": ("_repr_html_", "__repr__"),
}
CONF
    PYOPENGL_PLATFORM=egl MUJOCO_GL=egl TORCHRL_TUTORIALS_FAST=1 "$trial_python" -m sphinx -b dummy \
        "$gallery_source" "${trial_dir}/dummy-build"

)

# Copy the gallery page, checksum, notebook, and matching images into the main
# checkout. Sphinx Gallery will reuse the page during the nightly docs pass.
python - "$gallery_source/tutorials" "$root_dir/docs/source/tutorials" <<'PY'
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
