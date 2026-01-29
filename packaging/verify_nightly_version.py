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

# Check that version is not the major version (0.9.0)
if re.match(r"^\d+\.\d+\.\d+$", version):
    raise ValueError(f"Version should not be the major version: {version}")

# Check that version matches date format (YYYY.M.D)
date_pattern = r"^\d{4}\.\d{1,2}\.\d{1,2}$"
if not re.match(date_pattern, version):
    raise ValueError(f"Version should match date format YYYY.M.D, got: {version}")

# Verify it's today's date
today = date.today()
expected_version = f"{today.year}.{today.month}.{today.day}"
if version != expected_version:
    raise ValueError(f"Version should be today date {expected_version}, got: {version}")

print(f"✓ Version {version} is correctly formatted as nightly date")

# Check that tensordict-nightly is installed (not stable tensordict)
try:
    import tensordict

    # Check if it's the nightly version by looking at the version
    tensordict_version = tensordict.__version__
    print(f"Checking tensordict version: {tensordict_version}")

    # Check if it's a nightly version (either date format or contains 'd' followed by date)
    if re.match(date_pattern, tensordict_version) or "d2025" in tensordict_version:
        print(f"✓ tensordict version {tensordict_version} appears to be nightly")
    else:
        # Check if it's a stable version that should not be used in nightly builds
        if tensordict_version.startswith("0.9."):
            raise ValueError(
                f"tensordict should be nightly, not stable version: {tensordict_version}"
            )
        print(
            f"⚠ tensordict version {tensordict_version} - please verify this is nightly"
        )

except ImportError:
    raise ValueError(
        "tensordict is not installed - nightly builds should include tensordict-nightly"
    )
