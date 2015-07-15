# TODO: Create conda packages?
pip install "pylru >=1.0.6"
pip install "dill >=0.2.2"
pip install "beautifulsoup4 >=4.3.2"
pip install "pyeasyga >=0.3.0"

# TODO: Install from PyPI when newer versions of these package are
# available (pymdptoolbox >4.0-b3, PyBrain >0.3).
pip install https://github.com/sawcordwell/pymdptoolbox/zipball/master
pip install https://github.com/pybrain/pybrain/zipball/master

python setup.py install
