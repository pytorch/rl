# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test runner for TorchRL tutorials.

This module discovers and runs all tutorial .py files as individual pytest tests,
enabling flaky test tracking through the JSON report output.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Get the root directory of the repository
ROOT_DIR = Path(__file__).parents[4]
TUTORIALS_DIR = ROOT_DIR / "tutorials" / "sphinx-tutorials"

# Tutorials that should be skipped (e.g., require special dependencies or are too slow)
SKIP_TUTORIALS = {
    # Add tutorial filenames here if they need to be skipped
    # "llm_browser.py",  # Example: requires browser dependencies
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


@pytest.fixture(scope="module")
def check_gpu():
    """Check if GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.parametrize("tutorial_path", get_tutorial_files(), ids=get_tutorial_ids())
def test_tutorial(tutorial_path: Path, check_gpu):
    """Run a tutorial file and verify it completes without error."""
    tutorial_name = tutorial_path.name

    # Skip GPU tutorials if no GPU available
    if tutorial_name in GPU_TUTORIALS and not check_gpu:
        pytest.skip(f"Skipping {tutorial_name}: requires GPU")

    # Set environment variables for the tutorial
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use non-interactive matplotlib backend
    env["WANDB_MODE"] = "disabled"  # Disable wandb logging
    env["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages for CUDA

    # Run the tutorial as a subprocess
    # We use compile + exec pattern similar to run_local.sh
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import sys
with open('{tutorial_path}') as f:
    source = f.read()
code = compile(source, '{tutorial_path}', 'exec')
exec(code)
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout per tutorial
        cwd=str(ROOT_DIR),
    )

    # Check for failure
    if result.returncode != 0:
        pytest.fail(
            f"Tutorial {tutorial_name} failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


if __name__ == "__main__":
    # Allow running this file directly for debugging
    pytest.main([__file__, "-v", "--tb=short"])
