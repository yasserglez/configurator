#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Adaptive configuration dialogs.

Attributes:
    __version__: The current version string.
"""

import os
import subprocess


def _get_version(version=None):  # overwritten by setup.py
    if version is None:
        pkg_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(pkg_dir, os.pardir))
        git_dir = os.path.join(src_dir, ".git")
        git_args = ("git", "--work-tree", src_dir, "--git-dir",
                    git_dir, "describe", "--tags", "--always")
        output = subprocess.check_output(git_args)
        version = output.decode("utf-8").strip()
        if version.rfind("-") >= 0:
            version = version[:version.rfind("-")]  # strip SHA1 hash
            version = version.replace("-", ".post")  # PEP 440 compatible
    return version

__version__ = _get_version()
