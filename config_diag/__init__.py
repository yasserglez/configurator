# -*- coding: utf-8 -*-

"""
Constructing Adaptive Configuration Dialogs using Crowd Data
"""

import os
import subprocess


def get_version():
    """Get the version number.
    
    Try to obtain the current version string dinamically from Git using 
    'git describe'. If that fails, fallback to an internal variable. The 
    latter is used in checked-out source code and it is set via a custom 
     'python setup.py sdist/build' command when the code is exported.
    
    Returns:
        The version string.
    """
    pkg_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(pkg_dir, os.pardir))
    git_dir = os.path.join(src_dir, ".git")
    try:
        # Try to get the  version string dynamically from Git.
        git_args = ("git", "--work-tree", src_dir, "--git-dir", git_dir, 
                    "describe", "--tags", "--dirty")
        with open(os.devnull, "w") as devnull:
            version = subprocess.check_output(git_args, stderr=devnull).strip()
    except subprocess.CalledProcessError:
        # Overwritten by a custom 'python setup.py sdist/build' command.
        version = None
    return version

__version__ = get_version()
