from setuptools import setup

import configurator


setup(name="configurator",
      version=configurator.__version__,
      url="https://github.com/yasserglez/configurator",
      description="Calculation of optimal configuration processes.",
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: Apache Software License",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      author="Yasser Gonzalez",
      author_email="contact@yassergonzalez.com",
      packages=[
          "configurator",
          "configurator.dialogs",
          "configurator.util"
      ],
      install_requires=[
          "numpy >= 1.9.1",
          "scipy >= 0.14.0",
          "pylru >= 1.0.6",
          "python-igraph >= 0.7.1",
          "dill >= 0.2.2",
          "beautifulsoup4 >= 4.3.2",
          "lxml >= 3.4.2",
          "pyeasyga >= 0.3.0",
          "pymdptoolbox",
          "PyBrain",
          "fann2",
          "fim",
      ],
      dependency_links=[
          "http://www.borgelt.net/src/pyfim.tar.gz#egg=fim",
      ])
