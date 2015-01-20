"""Configurator: Adaptive Configuration Dialogs"""

import os
import subprocess


def get_version(version=None):  # overwritten by setup.py
    """Get the version string.

    Returns:
        The version string.
    """
    if version is None:
        pkg_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(pkg_dir, os.pardir))
        git_dir = os.path.join(src_dir, ".git")
        git_args = ("git", "--work-tree", src_dir, "--git-dir", git_dir,
                    "describe", "--tags", "--dirty")
        with open(os.devnull, "w") as devnull:
            output = subprocess.check_output(git_args, stderr=devnull)
            version = output.decode("utf-8").strip()
    return version

__version__ = get_version()
