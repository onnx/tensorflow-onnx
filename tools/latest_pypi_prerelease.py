# SPDX-License-Identifier: Apache-2.0

"""Find the latest usable PyPI prerelease newer than the current stable release."""

import argparse
import json

from packaging.version import InvalidVersion, Version


def latest_prerelease(releases):
    """Return the newest non-yanked prerelease newer than the latest stable."""
    stable_versions = []
    prerelease_versions = []

    for value, files in releases.items():
        try:
            version = Version(value)
        except InvalidVersion:
            continue

        if not files or all(file.get("yanked", False) for file in files):
            continue

        if version.is_prerelease and not version.is_devrelease:
            prerelease_versions.append(version)
        elif not version.is_devrelease:
            stable_versions.append(version)

    latest_stable = max(stable_versions, default=None)
    candidates = [
        version
        for version in prerelease_versions
        if latest_stable is None or version > latest_stable
    ]
    return str(max(candidates)) if candidates else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=argparse.FileType(encoding="utf-8"))
    args = parser.parse_args()
    print(latest_prerelease(json.load(args.metadata)["releases"]))


if __name__ == "__main__":
    main()
