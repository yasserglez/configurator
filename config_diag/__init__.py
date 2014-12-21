# -*- coding: utf-8 -*-

import os
import subprocess


def get_version():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    git_dir = os.path.join(src_dir, ".git")
    try:
        # Try to get the latest version string dynamically from Git:
        # (latest version, commits since latest release, and commit SHA-1)
        git_args = ("git", "--work-tree", src_dir, "--git-dir", git_dir, 
                    "describe", "--tags", "--dirty")
        with open(os.devnull, "w") as devnull:
            version = subprocess.check_output(git_args, stderr=devnull).strip()
    except subprocess.CalledProcessError:
        # Overwritten by custom 'python setup.py sdist/build' commands
        version = None
    return version

__version__ = get_version()
