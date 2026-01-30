# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test runner for TorchRL tutorials.

This module discovers and runs all tutorial .py files as individual pytest tests,
enabling flaky test tracking through the JSON report output.

Each tutorial runs in a subprocess with a 5-minute timeout.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Get the root directory of the repository
ROOT_DIR = Path(__file__).parents[4]
TUTORIALS_DIR = ROOT_DIR / "tutorials" / "sphinx-tutorials"

# Tutorials that should be skipped
SKIP_TUTORIALS = {
    "llm_browser.py",  # Requires transformers, browser dependencies
    "export.py",  # Requires PyTorch 2.6+ for aoti_compile_and_package
}

# Tutorials that require GPU
GPU_TUTORIALS = {
    "coding_ddpg.py",
    "coding_dqn.py",
    "coding_ppo.py",
    "dqn_with_rnn.py",
    "multiagent_competitive_ddpg.py",
    "multiagent_ppo.py",
    "pendulum.py",
}


def get_tutorial_files():
    """Discover all tutorial .py files."""
    tutorials = []
    for path in sorted(TUTORIALS_DIR.glob("*.py")):
        name = path.name
        if name.startswith("_"):
            continue
        if name in SKIP_TUTORIALS:
            continue
        tutorials.append(path)
    return tutorials


def get_tutorial_ids():
    """Get test IDs for parametrization."""
    return [p.stem for p in get_tutorial_files()]


def _has_gpu():
    """Check if GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


HAS_GPU = _has_gpu()


@pytest.mark.parametrize("tutorial_path", get_tutorial_files(), ids=get_tutorial_ids())
def test_tutorial(tutorial_path: Path):
    """Run a tutorial file and verify it completes without error.

    Each tutorial runs in its own subprocess with a 5-minute timeout.
    """
    tutorial_name = tutorial_path.name

    # Skip GPU tutorials if no GPU available
    if tutorial_name in GPU_TUTORIALS and not HAS_GPU:
        pytest.skip(f"Skipping {tutorial_name}: requires GPU")

    # Set environment variables for the tutorial
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["WANDB_MODE"] = "disabled"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["SDL_VIDEODRIVER"] = "dummy"

    # Run the tutorial as a subprocess with timeout
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import sys
import warnings
warnings.filterwarnings('ignore')
with open('{tutorial_path}') as f:
    source = f.read()
code = compile(source, '{tutorial_path}', 'exec')
exec(code)
""",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout per tutorial
            cwd=str(ROOT_DIR),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Tutorial {tutorial_name} timed out after 5 minutes")

    # Check for failure - keep output concise
    if result.returncode != 0:
        # Truncate output to last 50 lines for readability
        stderr_lines = result.stderr.strip().split("\n")
        stderr_tail = (
            "\n".join(stderr_lines[-50:]) if len(stderr_lines) > 50 else result.stderr
        )
        pytest.fail(
            f"Tutorial {tutorial_name} failed (exit code {result.returncode})\n"
            f"STDERR (last 50 lines):\n{stderr_tail}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
