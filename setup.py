# SPDX-License-Identifier: Apache-2.0


import distutils.command.build
import os
import subprocess
from collections import namedtuple
from textwrap import dedent

import setuptools.command.build_py
import setuptools.command.develop
from setuptools import Command, setup

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'tf2onnx')

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )


class create_version(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        alv2_license = '# SPDX-License-Identifier: Apache-2.0\n\n'
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(alv2_license + dedent('''
            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        setuptools.command.build_py.build_py.run(self)


class build(distutils.command.build.build):
    def run(self):
        self.run_command('build_py')


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        self.run_command('build')
        setuptools.command.develop.develop.run(self)


setup(
    version=VersionInfo.version,
    cmdclass={
        'create_version': create_version,
        'build_py': build_py,
        'build': build,
        'develop': develop,
    },
)
