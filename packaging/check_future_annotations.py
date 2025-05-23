# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import sys


def add_future_import(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    # Check if the import is already present
    for line in lines:
        if line.strip() == "from __future__ import annotations":
            return  # Import already present, no need to modify

    # Find the position to insert the import
    insert_pos = 0
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith("#"):
            insert_pos = i
            break

    # Insert the import statement after the first comment block
    lines.insert(insert_pos, "from __future__ import annotations\n\n")

    # Write the modified lines back to the file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    files = sys.argv[1:]
    for f in files:
        add_future_import(f)
    print("Processed files to ensure `from __future__ import annotations` is present.")


if __name__ == "__main__":
    main()
