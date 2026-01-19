import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd, *, cwd, env=None, timeout=60 * 60) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"cwd: {cwd}\n"
            f"cmd: {cmd}\n"
            f"exit_code: {proc.returncode}\n"
            f"output:\n{proc.stdout}"
        )
    return proc.stdout


def _pip_uninstall(pkg: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
        cwd=str(_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _install_cmd_prefix(installer: str) -> list[str]:
    if installer == "uv":
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _git(args) -> str:
    return _run(["git", *args], cwd=_ROOT, timeout=60).strip()


def _expected_dist_version(base_version: str) -> str:
    # Match the same release-branch rule as setup.py.
    branch = None
    for key in ("GITHUB_REF_NAME", "GIT_BRANCH", "BRANCH_NAME", "CI_COMMIT_REF_NAME"):
        val = os.environ.get(key)
        if val:
            branch = val
            break
    if branch is None:
        b = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        branch = None if b == "HEAD" else b
    if branch is not None and branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/") :]

    if branch is not None and (
        branch == f"release/v{base_version}"
        or branch.endswith(f"/release/v{base_version}")
    ):
        return base_version

    return f"{base_version}+g{_git(['rev-parse', '--short', 'HEAD'])}"


@pytest.mark.parametrize("editable", [True, False], ids=["editable", "wheel"])
def test_install_no_deps_has_nonzero_version(editable: bool, tmp_path: Path):
    # Requires git checkout.
    if not (_ROOT / ".git").exists():
        pytest.skip("not a git checkout")

    base_version = (_ROOT / "version.txt").read_text().strip()
    if not base_version:
        raise RuntimeError("Empty version.txt")

    expected = _expected_dist_version(base_version)

    installer = "uv" if shutil.which("uv") is not None else "pip"

    # Ensure we cover the historical failure mode where version becomes 0.0.0 when
    # build requirements aren't present (e.g. --no-build-isolation). This dedicated
    # CI job intentionally runs with --no-deps installs.
    _pip_uninstall("setuptools_scm")

    # Ensure clean re-install for each case.
    _pip_uninstall("torchrl")

    cmd = _install_cmd_prefix(installer)
    cmd.append("--no-deps")
    cmd.append("--no-build-isolation")
    if editable:
        cmd.extend(["-e", "."])
    else:
        cmd.append(".")

    _run(cmd, cwd=_ROOT, timeout=60 * 60)

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    code = r"""
import importlib.metadata as md
import json

out = {}
out["dist_version"] = md.version("torchrl")
try:
    import torchrl
    out["pkg_version"] = getattr(torchrl, "__version__", None)
    out["pkg_file"] = getattr(torchrl, "__file__", None)
except Exception as err:
    out["import_error"] = repr(err)

print(json.dumps(out))
"""
    out = _run([sys.executable, "-c", code], cwd=probe_dir, timeout=5 * 60)
    info = json.loads(out.strip())

    dist_version = str(info["dist_version"]).strip()
    assert dist_version != "0.0.0"
    assert dist_version == expected

    pkg_version = info.get("pkg_version")
    pkg_file = info.get("pkg_file")
    if pkg_version is not None and pkg_file is not None:
        pkg_version = str(pkg_version).strip()
        assert pkg_version != "0.0.0"
        assert pkg_version == expected

        pkg_path = Path(pkg_file).resolve()
        if editable:
            assert str(pkg_path).startswith(str(_ROOT.resolve()))
        else:
            assert "site-packages" in str(pkg_path)
    else:
        # If some hard dependency is missing, import can fail. The packaging version
        # should still be correct.
        assert "dist_version" in info
