#!/usr/bin/env python3

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize a ``.tar.gz`` sdist for byte-for-byte reproducible builds."""

import argparse
import gzip
import os
import pathlib
import tarfile
import tempfile


def normalize_sdist(source: pathlib.Path, epoch: int) -> None:
    """Rewrite *source* with stable ownership, permissions, and timestamps."""
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        with tempfile.NamedTemporaryFile(dir=source.parent, delete=False) as output:
            temporary = pathlib.Path(output.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as normalized:
                    for member in members:
                        contents = archive.extractfile(member) if member.isfile() else None
                        member.mtime = epoch
                        member.uid = 0
                        member.gid = 0
                        member.uname = ""
                        member.gname = ""
                        member.pax_headers = {
                            key: value
                            for key, value in member.pax_headers.items()
                            if key not in {"atime", "ctime", "gid", "gname", "mtime", "uid", "uname"}
                        }
                        if member.isdir():
                            member.mode = 0o755
                        elif member.isfile():
                            member.mode = 0o755 if member.mode & 0o111 else 0o644
                        normalized.addfile(member, contents)
    os.replace(temporary, source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=pathlib.Path)
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    parser.add_argument(
        "--epoch",
        type=int,
        default=int(source_date_epoch) if source_date_epoch is not None else None,
        help="Unix timestamp (defaults to SOURCE_DATE_EPOCH)",
    )
    args = parser.parse_args()
    if args.epoch is None:
        parser.error("--epoch or SOURCE_DATE_EPOCH is required")
    normalize_sdist(args.sdist, args.epoch)


if __name__ == "__main__":
    main()
