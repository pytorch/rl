# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys

HEADER = """# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""


def check_header(filename):
    with open(filename, encoding="utf-8") as f:
        file_content = f.read()

    if not file_content.startswith(HEADER):
        print(f"Missing or incorrect header in {filename}")
        return False
    return True


def main():
    files = sys.argv[1:]
    all_passed = True
    for f in files:
        if not check_header(f):
            all_passed = False
    if not all_passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
