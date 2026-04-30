#!/usr/bin/env python3
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_nightly_build():
    """Prepare pyproject.toml for nightly builds by modifying dependencies."""
    is_nightly = os.getenv("TORCHRL_NIGHTLY") == "1"

    if not is_nightly:
        logger.info("Not a nightly build, skipping pyproject.toml modification")
        return

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        logger.info("pyproject.toml not found")
        return

    # Read the current pyproject.toml
    with open(pyproject_path) as f:
        content = f.read()

    # Replace tensordict dependency with tensordict-nightly using regex
    # This pattern matches "tensordict" followed by any version constraints
    tensordict_pattern = r"tensordict[^,\]]*"
    if re.search(tensordict_pattern, content):
        content = re.sub(tensordict_pattern, "tensordict-nightly", content)
        logger.info("Replaced tensordict with tensordict-nightly in pyproject.toml")
    else:
        logger.info("tensordict dependency not found in pyproject.toml")

    # Write the modified content back
    with open(pyproject_path, "w") as f:
        f.write(content)

    logger.info("pyproject.toml prepared for nightly build")


if __name__ == "__main__":
    prepare_nightly_build()
