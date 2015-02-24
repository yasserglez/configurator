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
                    git_dir, "describe", "--tags")
        output = subprocess.check_output(git_args)
        output = output.decode("utf-8").strip()
        version = output[:output.rfind("-")]
    return version

__version__ = _get_version()
