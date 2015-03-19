import os
from distutils.command.sdist import sdist
from distutils.command.build import build
from setuptools import setup

import configurator


version = configurator.__version__

src_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(src_dir, "README.rst")) as f:
    long_description = f.read()


def update_version(init_py_file):
    # Save the version number in configurator/__init__.py
    with open(init_py_file) as f:
        init_py = f.read()
    os.unlink(init_py_file)  # might be a hard link
    init_py = init_py.replace("version=None", 'version="{0}"'.format(version))
    with open(init_py_file, "w") as f:
        f.write(init_py)


# Overwrite to call update_version
class custom_sdist(sdist):

    def make_release_tree(self, base_dir, files):
        sdist.make_release_tree(self, base_dir, files)
        init_py_file = os.path.join(base_dir, "configurator/__init__.py")
        update_version(init_py_file)


# Overwrite to call update_version
class custom_build(build):

    def run(self):
        build.run(self)
        init_py_file = os.path.join(self.build_lib, "configurator/__init__.py")
        update_version(init_py_file)


setup(name="configurator",
      version=version,
      description="Adaptive configuration dialogs.",
      long_description=long_description,
      author="Yasser Gonzalez",
      author_email="yasserglez@gmail.com",
      url="https://github.com/yasserglez/configurator",
      packages=["configurator"],
      install_requires=[
          "numpy >= 1.9.1",
          "scipy >= 0.14.0",
          "pylru >= 1.0.6",
          "python-igraph >= 0.7",
          "fann2",
          "pymdptoolbox",
          "PyBrain",
          "fim",
      ],
      dependency_links=[
          "http://www.borgelt.net/src/pyfim.tar.gz#egg=fim",
      ],
      classifiers=[
          "Programming Language :: Python :: 3 :: Only",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      cmdclass={"sdist": custom_sdist, "build": custom_build})
