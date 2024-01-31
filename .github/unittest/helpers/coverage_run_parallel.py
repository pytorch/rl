# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
When coverage run is launched with --concurrency=multiprocessing, it
needs a config file for further arguments.

This script is a drop-in wrapper to conveniently pass command line arguments
nevertheless. It writes temporary coverage config files on the fly and
invokes coverage with proper arguments
"""
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def write_config(config_path: Path, argv: List[str]) -> None:
    """
    Write a coverage.py config that is equivalent to the command line arguments passed here.
    Args:
        config_path: Path to write config to
        argv: Arguments passed to this script, which need to be converted to config file entries
    """
    assert not config_path.exists(), "Temporary coverage config exists already"
    cmdline = " ".join(shlex.quote(arg) for arg in argv[1:])
    with open(str(config_path), "wt", encoding="utf-8") as fh:
        fh.write(
            f"""# .coveragerc to control coverage.py
[run]
parallel=True
concurrency=
    multiprocessing
    thread
command_line={cmdline}
"""
        )


def main(argv: List[str]) -> int:
    if len(argv) < 1:
        print(  # noqa
            "Usage: 'python coverage_run_parallel.py <command> [command arguments]'"
        )
        sys.exit(1)
    # The temporary config is written into a temp dir that will be deleted
    # including all contents on context exit.
    # Note: Do not use tempfile.NamedTemporaryFile as this file cannot
    # be read by other processes on Windows
    with tempfile.TemporaryDirectory(prefix=".torchrl_coverage_config_tmp_") as tempdir:
        config_path = Path(tempdir) / ".coveragerc"
        os.environ["COVERAGE_RCFILE"] = str(
            config_path
        )  # This gets passed down to subprocesses
        write_config(config_path, argv)
        return subprocess.run(["coverage", "run"]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
