#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys

from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return (success, output)."""
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, cwd=cwd, text=True
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_black(files):
    """Run black formatter."""
    cmd = ["black", "--config", "pyproject.toml"] + files
    return run_command(cmd)


def run_ruff(files, fix=False):
    """Run ruff for fast linting."""
    cmd = ["ruff", "check"]
    if fix:
        cmd.append("--fix")
    cmd.extend(files)
    return run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Fast parallel linting")
    parser.add_argument(
        "files", nargs="*", help="Files to lint (default: all Python files)"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix issues where possible"
    )
    args = parser.parse_args()

    # If no files specified, find all Python files
    if not args.files:
        args.files = [
            str(p)
            for p in Path(".").rglob("*.py")
            if not any(part.startswith(".") for part in p.parts)
        ]

    if not args.files:
        print("No Python files found to lint")
        return 0

    # Run linters in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        # Black formatting
        futures.append(executor.submit(run_black, args.files))

        # Ruff (replaces flake8, pyupgrade, autoflake, pydocstyle, and usort)
        futures.append(executor.submit(run_ruff, args.files, args.fix))

        # Collect results
        success = True
        for future in concurrent.futures.as_completed(futures):
            ok, output = future.result()
            if output.strip():
                print(output)
            success = success and ok

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
