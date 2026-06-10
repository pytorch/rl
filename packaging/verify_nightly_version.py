#!/usr/bin/env python3
import re
import sys
from datetime import date

# Try to import torchrl with better error handling
try:
    import torchrl

    print(f"Successfully imported torchrl from: {torchrl.__file__}")
except ImportError as e:
    print(f"Failed to import torchrl: {e}")
    sys.exit(1)

# Check if __version__ attribute exists
if not hasattr(torchrl, "__version__"):
    print(
        f'Available attributes in torchrl: {[attr for attr in dir(torchrl) if not attr.startswith("_")]}'
    )
    raise AttributeError("torchrl module has no __version__ attribute")

version = torchrl.__version__
print(f"Checking version: {version}")

date_pattern = r"^\d{4}\.\d{1,2}\.\d{1,2}$"
if not isinstance(version, str) or not version:
    # A missing version must be fatal: when __version__ was None this check
    # silently passed while PyPI rejected the wheels.
    raise ValueError(
        "torchrl.__version__ is missing or empty; nightly wheels must expose "
        "a YYYY.M.D version string."
    )

# Check that version matches date format (YYYY.M.D). The 4-digit-year
# requirement rejects stable versions (e.g. 0.13.0), and the anchored pattern
# rejects PEP 440 local version identifiers (e.g. 2026.6.9+g<sha>), which
# PyPI refuses.
if not re.match(date_pattern, version):
    raise ValueError(f"Version should match date format YYYY.M.D, got: {version}")

# Verify it's today's date
today = date.today()
expected_version = f"{today.year}.{today.month}.{today.day}"
if version != expected_version:
    raise ValueError(f"Version should be today date {expected_version}, got: {version}")

print(f"Version {version} is correctly formatted as nightly date")

# Check that tensordict-nightly is installed (not stable tensordict)
try:
    import tensordict

    # Check if it's the nightly version by looking at the version
    tensordict_version = tensordict.__version__
    print(f"Checking tensordict version: {tensordict_version}")

    # Check if it's a nightly version (either date format or contains 'd' followed by date)
    if not isinstance(tensordict_version, str) or not tensordict_version:
        print(
            "WARNING: tensordict.__version__ is missing; skipping nightly "
            "version format validation."
        )
    elif re.match(date_pattern, tensordict_version) or re.search(
        r"d20\d{6}", tensordict_version
    ):
        print(f"tensordict version {tensordict_version} appears to be nightly")
    else:
        # Check if it's a stable version that should not be used in nightly builds
        if tensordict_version.startswith("0.9."):
            raise ValueError(
                f"tensordict should be nightly, not stable version: {tensordict_version}"
            )
        print(
            f"WARNING: tensordict version {tensordict_version} - please verify "
            "this is nightly"
        )

except ImportError:
    raise ValueError(
        "tensordict is not installed - nightly builds should include tensordict-nightly"
    )
