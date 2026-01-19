from __future__ import annotations

import contextlib
import glob
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.resolve()


def _check_pybind11():
    """Check that pybind11 is installed and provide a clear error message if not.

    Only checks when actually building extensions, not for commands like 'clean'.
    """
    # Commands that don't require building C++ extensions
    skip_commands = {"clean", "egg_info", "sdist", "--version", "--help", "-h"}
    if skip_commands.intersection(sys.argv):
        return
    if importlib.util.find_spec("pybind11") is None:
        raise RuntimeError(
            "pybind11 is required to build TorchRL's C++ extensions but was not found.\n"
            "Please install it with:\n"
            "    pip install 'pybind11[global]'\n"
            "Then re-run the installation."
        )


_check_pybind11()
_RELEASE_BRANCH_RE = re.compile(r"^release/v(?P<release_id>.+)$")
_BUILD_INFO_FILE = ROOT_DIR / "build" / ".torchrl_build_info.json"


def _check_and_clean_stale_builds():
    """Check if existing build was made with a different PyTorch version and clean if so.

    This prevents ABI incompatibility issues when switching between PyTorch versions.
    """
    current_torch_version = torch.__version__
    current_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    if _BUILD_INFO_FILE.exists():
        try:
            with open(_BUILD_INFO_FILE) as f:
                build_info = json.load(f)
            old_torch = build_info.get("torch_version")
            old_python = build_info.get("python_version")

            if (
                old_torch != current_torch_version
                or old_python != current_python_version
            ):
                logger.warning(
                    f"Detected PyTorch/Python version change: "
                    f"PyTorch {old_torch} -> {current_torch_version}, "
                    f"Python {old_python} -> {current_python_version}. "
                    f"Cleaning stale build artifacts..."
                )
                # Clean stale extension files for current Python version
                ext = ".pyd" if sys.platform == "win32" else ".so"
                ext_pattern = (
                    ROOT_DIR
                    / "torchrl"
                    / f"_torchrl.cpython-{sys.version_info.major}{sys.version_info.minor}*{ext}"
                )
                for so_file in glob.glob(str(ext_pattern)):
                    logger.warning(f"Removing stale: {so_file}")
                    os.remove(so_file)
                # Clean build directory
                build_dir = ROOT_DIR / "build"
                if build_dir.exists():
                    import shutil

                    for item in build_dir.iterdir():
                        if item.name.startswith("temp.") or item.name.startswith(
                            "lib."
                        ):
                            logger.warning(f"Removing stale build dir: {item}")
                            shutil.rmtree(item)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read build info: {e}")

    # Write current build info
    _BUILD_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_BUILD_INFO_FILE, "w") as f:
        json.dump(
            {
                "torch_version": current_torch_version,
                "python_version": current_python_version,
            },
            f,
        )


def get_extensions():
    """Build C++ extensions with platform-specific compiler flags.

    This function configures the C++ extension build process with appropriate
    compiler flags for different platforms:
    - Windows (MSVC): Uses /O2, /std:c++17, /EHsc flags
    - Unix-like (GCC/Clang): Uses -O3, -std=c++17, -fdiagnostics-color=always flags

    Returns:
        list: List of CppExtension objects to be built
    """
    extension = CppExtension
    extra_link_args = []

    # Platform-specific compiler flags
    if sys.platform == "win32":
        # MSVC flags for Windows
        extra_compile_args = {
            "cxx": [
                "/O2",  # Optimization level 2 (equivalent to -O3)
                "/std:c++17",  # C++17 standard
                "/EHsc",  # Exception handling model
            ]
        }
        debug_mode = os.getenv("DEBUG", "0") == "1"
        if debug_mode:
            logging.info("Compiling in debug mode")
            extra_compile_args = {
                "cxx": [
                    "/Od",  # No optimization (equivalent to -O0)
                    "/Zi",  # Generate debug info
                    "/std:c++17",  # C++17 standard
                    "/EHsc",  # Exception handling model
                ]
            }
            extra_link_args = ["/DEBUG"]
    else:
        # GCC/Clang flags for Unix-like systems
        extra_compile_args = {
            "cxx": [
                "-O3",
                "-std=c++17",
                "-fdiagnostics-color=always",
            ]
        }
        debug_mode = os.getenv("DEBUG", "0") == "1"
        if debug_mode:
            logging.info("Compiling in debug mode")
            extra_compile_args = {
                "cxx": [
                    "-O0",
                    "-fno-inline",
                    "-g",
                    "-std=c++17",
                    "-fdiagnostics-color=always",
                ]
            }
            extra_link_args = ["-O0", "-g"]

    extensions_dir = "torchrl/csrc"

    # Get just the filenames, not full paths
    cpp_files = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    sources = [os.path.relpath(f) for f in cpp_files]

    include_dirs = ["."]
    python_include_dir = os.getenv("PYTHON_INCLUDE_DIR")
    if python_include_dir is not None:
        include_dirs.append(python_include_dir)
    ext_modules = [
        extension(
            "torchrl._torchrl",
            sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


def _git_output(args) -> str | None:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=str(ROOT_DIR))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _branch_name() -> str | None:
    for key in (
        "GITHUB_REF_NAME",
        "GIT_BRANCH",
        "BRANCH_NAME",
        "CI_COMMIT_REF_NAME",
    ):
        val = os.environ.get(key)
        if val:
            return val
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if not branch or branch == "HEAD":
        return None
    return branch


def _short_sha() -> str | None:
    return _git_output(["rev-parse", "--short", "HEAD"])


def _version_with_local_sha(base_version: str) -> str:
    # Do not append local version on the matching release branch.
    branch = _branch_name()
    if branch:
        m = _RELEASE_BRANCH_RE.match(branch)
        if m and m.group("release_id").strip() == base_version.strip():
            return base_version
    sha = _short_sha()
    if not sha:
        return base_version
    return f"{base_version}+g{sha}"


@contextlib.contextmanager
def set_version():
    # Prefer explicit build version if provided by build tooling.
    if "SETUPTOOLS_SCM_PRETEND_VERSION" not in os.environ:
        override = os.environ.get("TORCHRL_BUILD_VERSION")
        if override:
            os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = override.strip()
        else:
            base_version = (ROOT_DIR / "version.txt").read_text().strip()
            full_version = _version_with_local_sha(base_version)
            os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = full_version
        yield
        del os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"]
        return
    yield


def main():
    """Main setup function for building TorchRL with C++ extensions."""
    # Check for stale builds from different PyTorch/Python versions
    _check_and_clean_stale_builds()

    with set_version():
        pretend_version = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION")
        _has_setuptools_scm = importlib.util.find_spec("setuptools_scm") is not None

        setup_kwargs = {
            "ext_modules": get_extensions(),
            "cmdclass": {"build_ext": BuildExtension.with_options()},
            "zip_safe": False,
            **(
                {"setup_requires": ["setuptools_scm"], "use_scm_version": True}
                if _has_setuptools_scm
                # pretend_version already includes +g<sha> (computed in set_version)
                else {"version": pretend_version}
            ),
        }

        setup(**setup_kwargs)


if __name__ == "__main__":
    main()
