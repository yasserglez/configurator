import os

from conftest import TESTS_DIR

from configurator.util import load_C2O, load_SXFM
from configurator.csp import CSP


def test_load_C2O():
    # http://www.jku.at/JKU_Site/JKU/isse/content/e139529/e126342/e126343/e266488/Car.xml
    xml_file = os.path.join(TESTS_DIR, "Car.xml")
    var_domains, constraints = load_C2O(xml_file)
    csp = CSP(var_domains, constraints)
    num_solutions = 0
    for solution in csp.solve():
        num_solutions += 1
    assert num_solutions == 168


def test_load_SXFM():
    # http://gsd.uwaterloo.ca:8088/SPLOT/models/REAL-FM-6.xml
    xml_file = os.path.join(TESTS_DIR, "REAL-FM-6.xml")
    var_domains, constraints = load_SXFM(xml_file)
    csp = CSP(var_domains, constraints)
    num_solutions = 0
    for solution in csp.solve():
        num_solutions += 1
    assert num_solutions == 14
