"""
Constructing Adaptive Configuration Dialogs using Crowd Data
"""

import os
import subprocess


def get_version():
    """Get the version string.

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


class ConfigDialog(object):
    """Adaptive configuration dialog.

    This is the base class of all the configuration dialogs defined in
    the package (not intented to be instantiated directly). It defines
    a common interface followed by the dialogs generated using the
    different ConfigDiagBuilder subclasses.
    """

    def __init__(self):
        """Initialize a new instance.
        """
        super().__init__()


class ConfigDialogBuilder(object):
    """Adaptive configuration dialog builder.

    Attributes:
        config_sample: A sample of the configuration variables.
    """

    def __init__(self, config_sample):
        """Initialize a new instance.

        Arguments:
            config_sample: A 2-dimensional numpy array containing a
                sample of the configuration variables.
        """
        super().__init__()
        self.config_sample = config_sample

    def build_dialog():
        """Construct the adaptive configuration dialog.

        Returns:
            An instance of a ConfigDialog subclass.
        """
        raise NotImplementedError()
