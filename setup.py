import contextlib
import glob
import importlib.util
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.resolve()
_RELEASE_BRANCH_RE = re.compile(r"^release/v(?P<release_id>.+)$")


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
