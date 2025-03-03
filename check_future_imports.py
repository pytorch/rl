# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os


def find_files_without_future_annotations(directory):
    """Finds Python files that do not contain 'from __future__ import annotations'."""
    files_without_annotations = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    if "from __future__ import annotations" not in content:
                        files_without_annotations.append(file_path)

    return files_without_annotations


if __name__ == "__main__":
    repo_directory = "."  # Change this to your repository's root directory
    files = find_files_without_future_annotations(repo_directory)
    if files:
        print("Files without 'from __future__ import annotations':")
        for file in files:
            print(file)
    else:
        print("All files contain 'from __future__ import annotations'.")
