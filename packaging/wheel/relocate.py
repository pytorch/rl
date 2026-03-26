# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Helper script to package wheels and relocate binaries."""

import glob
import hashlib

# Standard library imports
import os
import os.path as osp
import shutil
import sys
import zipfile
from base64 import urlsafe_b64encode

HERE = osp.dirname(osp.abspath(__file__))
PACKAGE_ROOT = osp.dirname(osp.dirname(HERE))


def rehash(path, blocksize=1 << 20):
    """Return (hash, length) for path using hashlib.sha256()"""
    h = hashlib.sha256()
    length = 0
    with open(path, "rb") as f:
        while block := f.read(blocksize):
            length += len(block)
            h.update(block)
    digest = "sha256=" + urlsafe_b64encode(h.digest()).decode("latin1").rstrip("=")
    # unicode/str python2 issues
    return (digest, str(length))  # type: ignore


def unzip_file(file, dest):
    """Decompress zip `file` into directory `dest`."""
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(dest)


def is_program_installed(basename):
    """
    Return program absolute path if installed in PATH.
    Otherwise, return None
    On macOS systems, a .app is considered installed if
    it exists.
    """
    if sys.platform == "darwin" and basename.endswith(".app") and osp.exists(basename):
        return basename

    for path in os.environ["PATH"].split(os.pathsep):
        abspath = osp.join(path, basename)
        if osp.isfile(abspath):
            return abspath


def find_program(basename):
    """
    Find program in PATH and return absolute path
    Try adding .exe or .bat to basename on Windows platforms
    (return None if not found)
    """
    names = [basename]
    if os.name == "nt":
        # Windows platforms
        extensions = (".exe", ".bat", ".cmd", ".dll")
        if not basename.endswith(extensions):
            names = [basename + ext for ext in extensions] + [basename]
    for name in names:
        path = is_program_installed(name)
        if path:
            return path


def compress_wheel(output_dir, wheel, wheel_dir, wheel_name):
    """Create RECORD file and compress wheel distribution."""
    # ("Update RECORD file in wheel")
    dist_info = glob.glob(osp.join(output_dir, "*.dist-info"))[0]
    record_file = osp.join(dist_info, "RECORD")

    with open(record_file, "w") as f:
        for root, _, files in os.walk(output_dir):
            for this_file in files:
                full_file = osp.join(root, this_file)
                rel_file = osp.relpath(full_file, output_dir)
                if full_file == record_file:
                    f.write(f"{rel_file},,\n")
                else:
                    digest, size = rehash(full_file)
                    f.write(f"{rel_file},{digest},{size}\n")

    # ("Compressing wheel")
    base_wheel_name = osp.join(wheel_dir, wheel_name)
    shutil.make_archive(base_wheel_name, "zip", output_dir)
    os.remove(wheel)
    shutil.move(f"{base_wheel_name}.zip", wheel)
    shutil.rmtree(output_dir)


def patch_win():
    # Get dumpbin location
    dumpbin = find_program("dumpbin")
    if dumpbin is None:
        raise FileNotFoundError(
            "Dumpbin was not found in the system, please make sure that is available on the PATH."
        )

    # Find wheel
    # ("Finding wheels...")
    wheels = glob.glob(osp.join(PACKAGE_ROOT, "dist", "*.whl"))
    output_dir = osp.join(PACKAGE_ROOT, "dist", ".wheel-process")

    for wheel in wheels:
        print(f"processing {wheel}")
        if osp.exists(output_dir):
            shutil.rmtree(output_dir)
        print(f"creating output directory {output_dir}")
        os.makedirs(output_dir)

        # ("Unzipping wheel...")
        wheel_file = osp.basename(wheel)
        wheel_dir = osp.dirname(wheel)
        # (f"{wheel_file}")
        wheel_name, _ = osp.splitext(wheel_file)
        print(f"unzipping {wheel} in {output_dir}")
        unzip_file(wheel, output_dir)
        print("compressing wheel")
        compress_wheel(output_dir, wheel, wheel_dir, wheel_name)


if __name__ == "__main__":
    if sys.platform == "linux":
        pass
    elif sys.platform == "win32":
        patch_win()
    else:
        raise NotImplementedError
