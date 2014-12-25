import os
from distutils.command.sdist import sdist
from distutils.command.build import build
from setuptools import setup, find_packages

import config_diag


version = config_diag.__version__

src_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(src_dir, "README.txt")) as f:
    long_description = f.read()


def update_version(init_py_file):
    # Save the version number in config_diag/__init__.py
    with open(init_py_file) as f:
        init_py = f.read()
    os.unlink(init_py_file)  # Might be a hard link
    init_py = init_py.replace("version = None", 'version = "{0}"'.format(version))
    with open(init_py_file, "w") as f:
        f.write(init_py)

class custom_sdist(sdist):
    # Call update_version
    def make_release_tree(self, base_dir, files):
        sdist.make_release_tree(self, base_dir, files)
        init_py_file = os.path.join(base_dir, "config_diag/__init__.py")
        update_version(init_py_file)

class custom_build(build):
    # Call update_version
    def run(self):
        build.run(self)
        init_py_file = os.path.join(self.build_lib, "config_diag/__init__.py")
        update_version(init_py_file)


setup(name="config_diag",
      version=version,
      description="Constructing Adaptive Configuration Dialogs using Crowd Data",
      long_description=long_description,
      author="Yasser Gonzalez",
      author_email="yasserglez@gmail.com",
      packages=find_packages(exclude=["*.tests"]),
      classifiers=[
          "Programming Language :: Python :: 3 :: Only",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Software Development :: User Interfaces",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      test_suite="nose.collector",
      cmdclass={"sdist": custom_sdist, "build": custom_build})
