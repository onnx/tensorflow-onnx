# SPDX-License-Identifier: Apache-2.0

import unittest

from tools.latest_pypi_prerelease import latest_prerelease


class LatestPrereleaseTest(unittest.TestCase):
    def test_returns_latest_usable_prerelease_newer_than_stable(self):
        releases = {
            "invalid": [{"yanked": False}],
            "1.21.0rc1": [{"yanked": False}],
            "1.22.0": [{"yanked": False}],
            "1.23.0.dev1": [{"yanked": False}],
            "1.23.0rc1": [{"yanked": True}],
            "1.23.0rc2": [{"yanked": False}],
            "1.23.0rc3": [],
        }

        self.assertEqual(latest_prerelease(releases), "1.23.0rc2")

    def test_returns_empty_when_only_stale_prereleases_exist(self):
        releases = {
            "1.21.0rc1": [{"yanked": False}],
            "1.21.0": [{"yanked": False}],
            "1.22.0rc1": [{"yanked": False}],
            "1.22.0": [{"yanked": False}],
        }

        self.assertEqual(latest_prerelease(releases), "")


if __name__ == "__main__":
    unittest.main()
