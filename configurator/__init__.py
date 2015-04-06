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
